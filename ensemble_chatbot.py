from keras_wrapper.dataset import loadDataset
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.utils import decode_predictions_beam_search
from config import load_parameters
from keras_wrapper.model_ensemble import BeamSearchEnsemble
import os

"""## 3. Decoding with a trained Neural Machine Translation Model

Now, we'll load from disk the model we just trained and we'll apply it for translating new text. In this case, we want to translate the 'test' split from our dataset.

Since we want to translate a new data split ('test') we must add it to the dataset instance, just as we did before (at the first tutorial). In case we also had the refences of the test split and we wanted to evaluate it, we can add it to the dataset. Note that this is not mandatory and we could just predict without evaluating.
"""
MODEL_PATH1 = os.path.join(os.getcwd(), 'models/empathy_100_hidden')
MODEL_PATH2 = os.path.join(os.getcwd(), 'models/persona_chat_lstm')

epoch_choice1 = 6
epoch_choice2 = 8

dataset1 = loadDataset(os.path.join(MODEL_PATH1, "dataset/Dataset_tutorial_dataset.pkl"))
dataset2 = loadDataset(os.path.join(MODEL_PATH1, "dataset/Dataset_tutorial_dataset.pkl"))

dataset2 = update_dataset_from_file(dataset2, args.text, params, splits=args.splits, remove_outputs=True)

# Load model
nmt_model1 = loadModel(MODEL_PATH1, epoch_choice1)
nmt_model2 = loadModel(MODEL_PATH2, epoch_choice2)


params = nmt_model1.params

# Define the inputs and outputs mapping from our Dataset instance to our model 
inputMapping = dict() 
for i, id_in in enumerate(params['INPUTS_IDS_DATASET']): 
    pos_source = dataset.ids_inputs.index(id_in) 
    id_dest = nmt_model.ids_inputs[i] 
    inputMapping[id_dest] = pos_source 
nmt_model.setInputsMapping(inputMapping) 

outputMapping = dict() 
for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']): 
    pos_target = dataset.ids_outputs.index(id_out) 
    id_dest = nmt_model.ids_outputs[i] 
    outputMapping[id_dest] = pos_target 
nmt_model.setOutputsMapping(outputMapping)

params_prediction = {
    'language': 'en',
    'tokenize_f': eval('dataset.' + 'tokenize_basic'),
    'beam_size': 20,
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
    'length_penalty' : True
  }

user_inputs = []
bot_responses = []
while True:
    user_input = input()
    if (user_input == 'exit()'):
        break
    user_inputs.append(user_input)
    with open(os.path.join(MODEL_PATH1, 'user_input.txt'), 'w') as f:
        f.write(user_input)
    dataset.setInput(os.path.join(MODEL_PATH1, 'user_input.txt'),
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

    dataset.setRawInput(os.path.join(MODEL_PATH1, 'user_input.txt'),
                  'test',
                  type='file-name',
                  id='raw_source_text',
                  overwrite_split=True)

    vocab = dataset.vocabulary['target_text']['idx2words']

    beam_searcher = BeamSearchEnsemble([nmt_model],
                                       dataset,
                                       params_prediction,
                                       n_best=False,
                                       verbose=1)
    predictions = beam_searcher.predictBeamSearchNet()['test']

    # n_best_predictions = []
    # for i, (n_best_preds, n_best_scores, n_best_alphas) in enumerate(predictions['n_best']):
    #     n_best_sample_score = []
    #     for n_best_pred, n_best_score, n_best_alpha in zip(n_best_preds, n_best_scores, n_best_alphas):
    #         pred = decode_predictions_beam_search([n_best_pred],
    #                                               vocab,
    #                                               # alphas=[n_best_alpha] if params_prediction['pos_unk'] else None,
    #                                               # x_text=[sources[i]] if params_prediction['pos_unk'] else None,
    #                                               verbose=1)
    #         n_best_sample_score.append([i, pred, n_best_score])
    #     n_best_predictions.append(n_best_sample_score)

    # print(n_best_predictions)
    # continue


    # predictions = nmt_model.predictBeamSearchNet(dataset, params_prediction)['test']

    if params_prediction['pos_unk']:
        samples = predictions['samples']
        alphas = predictions['alphas']
    else:
        samples = predictions
        heuristic = None
        sources = None

    predictions = decode_predictions_beam_search(samples, vocab, verbose=params['VERBOSE'])
    
    print(predictions)
    bot_responses.append(predictions[0])


with open(os.path.join(MODEL_PATH1, 'test_conversation_epoch' + str(epoch_choice) + '.txt'), mode='w+') as file:
    for i in range(len(user_inputs)):
        file.write('user: ' + user_inputs[i] + '\n')
        file.write('bot: ' + bot_responses[i] + '\n')
    file.close()