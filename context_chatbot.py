from keras_wrapper.dataset import loadDataset
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.utils import decode_predictions_beam_search
from config import load_parameters
from keras_wrapper.extra.read_write import list2file
import os
import re

"""## 3. Decoding with a trained Neural Machine Translation Model

Now, we'll load from disk the model we just trained and we'll apply it for translating new text. In this case, we want to translate the 'test' split from our dataset.

Since we want to translate a new data split ('test') we must add it to the dataset instance, just as we did before (at the first tutorial). In case we also had the refences of the test split and we wanted to evaluate it, we can add it to the dataset. Note that this is not mandatory and we could just predict without evaluating.
"""
DATA_PATH = os.path.join(os.getcwd(), 'data/PersonaChat/')
MODEL_PATH = os.path.join(os.getcwd(), 'models/persona_chat_context_lstm_13_de_layers')

dataset = loadDataset(os.path.join(MODEL_PATH, "dataset/Dataset_tutorial_dataset.pkl"))

epoch_choice = 17
# Load model
nmt_model = loadModel(MODEL_PATH, epoch_choice)

params = load_parameters()

params_prediction = {
    'language': 'en',
    'tokenize_f': eval('dataset.' + 'tokenize_basic'),
    'beam_size': 6,
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
    'heuristic': 0,
    'state_below_maxlen': -1,
    'predict_on_sets': ['test'],
    'verbose': 0,
  }

user_inputs = []
bot_responses = []
context = list()
while True:
    user_input = input()
    if (user_input == 'exit()'):
    	break
    if (not re.search(r'[.!?]', user_input[-1])):
        user_input += '.'
    user_inputs.append(user_input)
    if (len(context) > 2):
        context.pop(0)
    context.append(user_input)
    context_string = ' '.join(context)
    print(context_string)
    with open(os.path.join(MODEL_PATH, 'context.txt'), 'w') as f:
        f.write(context_string)
    dataset.setInput(os.path.join(MODEL_PATH, 'context.txt'),
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

    dataset.setRawInput(os.path.join(DATA_PATH, 'context.txt'),
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

    predictions = decode_predictions_beam_search(samples,  vocab, verbose=params['VERBOSE'])
    
    print(predictions[0])
    bot_responses.append(predictions[0])
    if (len(context) > 2):
        context.pop(0)
    context.append(predictions[0])
    # text = open(os.path.join(DATA_PATH, 'train_y.txt')).read()
    # lines = text.split('\n')
    # for i, line in enumerate(lines):
    #     print('y_true: ' + line + '\t\ty_pred: ' + predictions[i])
    # filepath = '/content/drive/My Drive/test/user_input_preds.txt'
    # list2file(filepath, predictions)
    # with open(filepath, 'r') as f:
    #    pred = f.readline()
    # print(pred)

with open(os.path.join(MODEL_PATH, 'test_conversation_e' + str(epoch_choice) + '.txt'), mode='w+') as file:
	for i in range(len(user_inputs)):
		file.write('user: ' + user_inputs[i] + '\n')
		file.write('bot: ' + bot_responses[i] + '\n')
	file.close()