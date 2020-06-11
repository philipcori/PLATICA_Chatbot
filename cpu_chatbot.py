from keras_wrapper.dataset import loadDataset
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.utils import decode_predictions_beam_search
from keras_wrapper.saving import saveModel
from config import load_parameters
from keras_wrapper.extra.read_write import list2file
import os
from keras_wrapper.cnn_model import updateModel
from nmt_keras.model_zoo import TranslationModel
from nmt_keras.build_callbacks import buildCallbacks


SRC_MODEL_PATH = os.path.join(os.getcwd(), 'models/persona_and_empathy_100_hidden')
DST_MODEL_PATH = os.path.join(os.getcwd(), 'models/persona_and_empathy_100_hidden_cpu')
epoch_choice = 18

dataset = loadDataset(os.path.join(SRC_MODEL_PATH, "dataset/Dataset_tutorial_dataset.pkl"))

src_model = loadModel(SRC_MODEL_PATH, epoch_choice)
params = src_model.params
params['USE_CUDNN'] = False
# params['BIDIRECTIONAL_ENCODER'] = False  # Set to False to get the RNN type displayed. 
params['MODEL_NAME'] = 'CPU'
params['STORE_PATH'] = DST_MODEL_PATH
params['MODE'] = 'sampling'
params['RELOAD'] = epoch_choice
# params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
# params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

cpu_model = TranslationModel(params,
                             model_type=params['MODEL_TYPE'],
                             verbose=True,
                             model_name=params['MODEL_NAME'],
                             vocabularies=dataset.vocabulary,
                             store_path=params['STORE_PATH'],
                             set_optimizer=True,
                             clear_dirs=True
                             )
exit()

cpu_model = updateModel(cpu_model, 
                        SRC_MODEL_PATH, 
                        params['RELOAD'], 
                        reload_epoch=True)

saveModel(cpu_model, update_num=epoch_choice, path=DST_MODEL_PATH, full_path=True)



# Define the inputs and outputs mapping from our Dataset instance to our model 
inputMapping = dict() 
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']): 
	pos_source = dataset.ids_inputs.index(id_in) 
	id_dest = cpu_model.ids_inputs[i] 
	inputMapping[id_dest] = pos_source 
cpu_model.setInputsMapping(inputMapping) 

outputMapping = dict() 
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']): 
	pos_target = dataset.ids_outputs.index(id_out) 
	id_dest = cpu_model.ids_outputs[i] 
	outputMapping[id_dest] = pos_target 
cpu_model.setOutputsMapping(outputMapping)

# callbacks = buildCallbacks(params, cpu_model, dataset)
# training_params['extra_callbacks']: callbacks
cpu_model.trainNet(dataset, params)

exit()

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
    'state_below_maxlen': 1,
    'predict_on_sets': ['test'],
    'verbose': 0,
  }

user_inputs = []
bot_responses = []
while True:
    user_input = input()
    if (user_input == 'exit()'):
    	break
    user_inputs.append(user_input)
    with open(os.path.join(DST_MODEL_PATH, 'user_input.txt'), 'w') as f:
        f.write(user_input)
    dataset.setInput(os.path.join(DST_MODEL_PATH, 'user_input.txt'),
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

    dataset.setRawInput(os.path.join(DST_MODEL_PATH, 'user_input.txt'),
                  'test',
                  type='file-name',
                  id='raw_source_text',
                  overwrite_split=True)

    
    vocab = dataset.vocabulary['target_text']['idx2words']
    predictions = cpu_model.predictBeamSearchNet(dataset, params_prediction)['test']

    if params_prediction['pos_unk']:
        samples = predictions['samples'] # The first element of predictions contain the word indices.
        alphas = predictions['alphas']
    else:
        samples = predictions
        heuristic = None
        sources = None

    predictions = decode_predictions_beam_search(samples, vocab, verbose=params['VERBOSE'])
    
    print(predictions[0])
    bot_responses.append(predictions[0])
    # text = open(os.path.join(MODEL_PATH, 'train_y.txt')).read()
    # lines = text.split('\n')
    # for i, line in enumerate(lines):
    #     print('y_true: ' + line + '\t\ty_pred: ' + predictions[i])
    # filepath = '/content/drive/My Drive/test/user_input_preds.txt'
    # list2file(filepath, predictions)
    # with open(filepath, 'r') as f:
    #    pred = f.readline()
    # print(pred)

with open(os.path.join(DST_MODEL_PATH, 'test_conversation_epoch' + str(epoch_choice) + '.txt'), mode='w+') as file:
	for i in range(len(user_inputs)):
		file.write('user: ' + user_inputs[i] + '\n')
		file.write('bot: ' + bot_responses[i] + '\n')
	file.close()