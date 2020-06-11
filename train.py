"""# NMT-Keras tutorial
---

This notebook describes, step by step, how to build a neural machine translation model with NMT-Keras. The tutorial is organized in different sections:


1. Create a Dataset instance, in order to properly manage the data. 
2. Create and train the Neural Translation Model in the training data.
3. Apply the trained model on new (unseen) data.
"""

# %tensorflow_version 1.x
import os
from keras_wrapper.dataset import Dataset, saveDataset
from data_engine.prepare_data import keep_n_captions
from keras_wrapper.utils import decode_predictions_beam_search
from keras_wrapper.extra.read_write import list2file
from config import load_parameters
from nmt_keras.model_zoo import TranslationModel
from nmt_keras.training import train_model
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.callbacks import PrintPerformanceMetricOnEpochEndOrEachNUpdates


DATA_PATH = os.path.join(os.getcwd(), 'data/EmpatheticDialogues/persona_and_empathy/')
GLOVE_PATH = os.path.join(os.getcwd(), 'data/glove.6B.100d/glove.6B.100d_pickle.pkl')           # note: delete .item() in model_zoo.py when loading
MODEL_PATH = os.path.join(os.getcwd(), 'models/persona_and_empathy_lstm_100_hidden_high_lr')           # set clear_dir to False in training.py


def start_training(use_gpu):

    ds = Dataset('tutorial_dataset', 'tutorial', silence=False)
    ds.setOutput(DATA_PATH + "train_y.txt",
                 'train',
                 type='text',
                 id='target_text',
                 tokenization='tokenize_basic',
                 build_vocabulary=True,
                 pad_on_batch=True,
                 sample_weights=True,
                 max_text_len=30,
                 max_words=30000,
                 min_occ=0)

    ds.setOutput(DATA_PATH + "val_y.txt",
                 'val',
                 type='text',
                 id='target_text',
                 pad_on_batch=True,
                 tokenization='tokenize_basic',
                 sample_weights=True,
                 max_text_len=30,
                 max_words=0)

    ds.setInput(DATA_PATH + "train_x.txt",
                'train',
                type='text',
                id='source_text',
                pad_on_batch=True,
                tokenization='tokenize_basic',
                build_vocabulary=True,
                fill='end',
                max_text_len=30,
                max_words=30000,
                min_occ=0)

    ds.setInput(DATA_PATH + "val_x.txt",
                'val',
                type='text',
                id='source_text',
                pad_on_batch=True,
                tokenization='tokenize_basic',
                fill='end',
                max_text_len=30,
                min_occ=0)

    ds.setInput(DATA_PATH + "train_y.txt",
                'train',
                type='text',
                id='state_below',
                required=False,
                tokenization='tokenize_basic',
                pad_on_batch=True,
                build_vocabulary='target_text',
                offset=1,
                fill='end',
                max_text_len=30,
                max_words=30000)
    
    ds.setInput(None,
                'val',
                type='ghost',
                id='state_below',
                required=False)

    for split, input_text_filename in zip(['train', 'val'], [DATA_PATH + "train_x.txt", DATA_PATH + "val_x.txt"]):
        ds.setRawInput(input_text_filename,
                      split,
                      type='file-name',
                      id='raw_source_text',
                      overwrite_split=True)

    """We also need to match the references with the inputs. Since we only have one reference per input sample, we set `repeat=1`."""

    keep_n_captions(ds, repeat=1, n=1, set_names=['val'])

    """Finally, we can save our dataset instance for using in other experiments:"""

    saveDataset(ds, MODEL_PATH + "/dataset")

    """## 2. Creating and training a Neural Translation Model
    Now, we'll create and train a Neural Machine Translation (NMT) model. Since there is a significant number of hyperparameters, we'll use the default ones, specified in the `config.py` file. Note that almost every hardcoded parameter is automatically set from config if we run  `main.py `.

    We'll create an `'AttentionRNNEncoderDecoder'` (a LSTM encoder-decoder with attention mechanism). Refer to the [`model_zoo.py`](https://github.com/lvapeab/nmt-keras/blob/master/nmt_keras/model_zoo.py) file for other models (e.g. Transformer). 

    So first, let's import the model and the hyperparameters. We'll also load the dataset we stored in the previous section (not necessary as it is in memory, but as a demonstration):
    """

    params = load_parameters()
    dataset = loadDataset(MODEL_PATH + "/dataset/Dataset_tutorial_dataset.pkl")

    """Since the number of words in the dataset may be unknown beforehand, we must update the params information according to the dataset instance:"""

    # PARAMETERS ACCORDING TO https://arxiv.org/pdf/1510.03055.pdf
    params['MODEL_TYPE'] = 'AttentionRNNEncoderDecoder'
    params['USE_CUDNN'] = use_gpu
    params['EARLY_STOP'] = True
    params['PATIENCE'] = 10
    params['SAVE_EACH_EVALUATION'] = True
    params['STORE_PATH'] = MODEL_PATH
    params['SOURCE_TEXT_EMBEDDING_SIZE'] = 32
    params['TARGET_TEXT_EMBEDDING_SIZE'] = 32
    params['SKIP_VECTORS_HIDDEN_SIZE'] = 32
    params['ATTENTION_SIZE'] = 32
    # params['SRC_PRETRAINED_VECTORS'] = GLOVE_PATH
    # params['TRG_PRETRAINED_VECTORS'] = GLOVE_PATH
    # params['SKIP_VECTORS_HIDDEN_SIZE'] = 100
    params['ENCODER_HIDDEN_SIZE'] = 32
    params['DECODER_HIDDEN_SIZE'] = 32
    params['N_LAYERS_ENCODER'] = 2
    params['N_LAYERS_DECODER'] = 2
    params['APPLY_DETOKENIZATION'] = True
    params['MAX_INPUT_TEXT_LEN'] = 24
    params['MAX_OUTPUT_TEXT_LEN'] = 24
    params['STOP_METRIC'] = 'perplexity'
    params['POS_UNK'] = True
    params['BEAM_SIZE'] = 20
    params['N_GPUS'] = 2
    params['START_EVAL_ON_EPOCH'] = 1
    params['BATCH_SIZE'] = 128
    params['EVAL_EACH'] = 1
    params['MAX_EPOCH'] = 50
    params['PLOT_EVALULATION'] = True
    params['APPLY_DETOKENIZATION'] = True
    params['MODE'] = 'training'
    params['BEAM_SEARCH'] = True
    params['TENSORBOARD'] = True
    # params['LR'] = 0.1
    train_model(params, load_dataset = MODEL_PATH + "/dataset/Dataset_tutorial_dataset.pkl")    


def resume_training(latest_epoch):
    params = load_parameters()
    params['RELOAD'] = latest_epoch
    params['MODEL_TYPE'] = 'AttentionRNNEncoderDecoder'
    params['USE_CUDNN'] = use_gpu
    params['EARLY_STOP'] = True
    params['PATIENCE'] = 10
    params['SAVE_EACH_EVALUATION'] = True
    params['STORE_PATH'] = MODEL_PATH
    params['SOURCE_TEXT_EMBEDDING_SIZE'] = 32
    params['TARGET_TEXT_EMBEDDING_SIZE'] = 32
    params['SKIP_VECTORS_HIDDEN_SIZE'] = 32
    params['ATTENTION_SIZE'] = 32
    params['ENCODER_HIDDEN_SIZE'] = 32
    params['DECODER_HIDDEN_SIZE'] = 32
    params['N_LAYERS_ENCODER'] = 4
    params['N_LAYERS_DECODER'] = 4
    params['APPLY_DETOKENIZATION'] = True
    params['MAX_INPUT_TEXT_LEN'] = 24
    params['MAX_OUTPUT_TEXT_LEN'] = 24
    params['STOP_METRIC'] = 'perplexity'
    params['POS_UNK'] = True
    params['BEAM_SIZE'] = 20
    params['N_GPUS'] = 2
    params['START_EVAL_ON_EPOCH'] = 1
    params['BATCH_SIZE'] = 256
    params['EVAL_EACH'] = 1
    params['MAX_EPOCH'] = 300
    params['PLOT_EVALULATION'] = True
    params['APPLY_DETOKENIZATION'] = True
    params['MODE'] = 'training'
    params['BEAM_SEARCH'] = True
    params['TENSORBOARD'] = True
    params['LR'] = 0.1
    train_model(params, load_dataset = MODEL_PATH + "/dataset/Dataset_tutorial_dataset.pkl")  

use_gpu = False

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
    try:
        os.mkdir(MODEL_PATH)  
    except OSError as error:
        print("Model directory already created.")  
    epoch_list = os.listdir(MODEL_PATH)
    i = 0
    latest_epoch = 0
    while i <= 500:
        if "".join(["epoch_", str(i), ".h5"]) in epoch_list:
            latest_epoch = i
        i += 1
    print('starting on epoch:' + str(latest_epoch))
    if latest_epoch == 0:
        start_training(use_gpu)
    elif latest_epoch != 0:
        resume_training(latest_epoch)

if __name__ == '__main__':
    main()