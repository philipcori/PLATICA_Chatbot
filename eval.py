from keras_wrapper.dataset import loadDataset
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.utils import decode_predictions_beam_search
from config import load_parameters
from keras_wrapper.extra.read_write import list2file
import os

# For using transformer set pos_unk to False, state_below_maxlen to -1.

DATA_PATH = os.path.join(os.getcwd(), 'data/EmpatheticDialogues/persona_and_empathy/')
MODEL_PATH = os.path.join(os.getcwd(), 'models/persona_and_empathy_100_hidden')

dataset = loadDataset(os.path.join(MODEL_PATH, "dataset/Dataset_tutorial_dataset.pkl"))

# Load model
nmt_model = loadModel(MODEL_PATH, 18)

params = nmt_model.params
params_prediction = {
    'language': 'en',
    'tokenize_f': eval('dataset.' + 'tokenize_basic'),
    'beam_size': 4,
    'optimized_search': True,
    'model_inputs': params['INPUTS_IDS_MODEL'],
    'model_outputs': params['OUTPUTS_IDS_MODEL'],
    'dataset_inputs':  params['INPUTS_IDS_DATASET'],
    'dataset_outputs':  params['OUTPUTS_IDS_DATASET'],
    'n_parallel_loaders': 1,
    'maxlen': 50,
    'model_inputs': ['source_text', 'state_below'],
    'model_outputs': ['target_text'],
    'dataset_inputs': ['source_text', 'state_below'],
    'dataset_outputs': ['target_text'],
    'normalize': True,
    'pos_unk': True,
    'attend_on_output' : True,
    'heuristic': 0,
    'state_below_maxlen': -1,
    'predict_on_sets': ['test'],
    'verbose': 0,
    'length_penalty' : True,
    'length_norm_penalty' : 0.8
  }


dataset.setInput(os.path.join(DATA_PATH, 'val_x.txt'),
        'test',
        type='text',
        id='source_text',
        pad_on_batch=True,
        tokenization='tokenize_basic',
        fill='end',
        max_text_len=30,
        min_occ=0,
        overwrite_split=True)

dataset.setInput(None,
            'test',
            type='ghost',
            id='state_below',
            required=False,
            overwrite_split=True)

dataset.setRawInput(os.path.join(DATA_PATH, 'val_x.txt'),
              'test',
              type='file-name',
              id='raw_source_text',
              overwrite_split=True)


vocab = dataset.vocabulary['target_text']['idx2words']
predictions = nmt_model.predictBeamSearchNet(dataset, params_prediction)['test']

if params_prediction['pos_unk']:
        samples = predictions['samples'] # The first element of predictions contain the word indices.
        alphas = predictions['alphas']
else:
    samples = predictions
    heuristic = None
    sources = None

predictions = decode_predictions_beam_search(samples, vocab, verbose=params['VERBOSE'])


with open(os.path.join(MODEL_PATH, 'val_y.txt'), mode='w+') as file:
    for pred in predictions:
        file.write(pred + '\n')
    file.close()