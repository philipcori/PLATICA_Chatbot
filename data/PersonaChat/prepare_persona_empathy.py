import json
import pandas as pd
import re
import random
val_rate = 0.001

# TODO: remove very common phrases, response has high n-gram overlap with context, 

def main():
	with open('personachat_self_original.json') as json_file:
		data = json.load(json_file)['train']
		conversations = [elt['utterances'][-1]['history'] for elt in data]
		contexts = []
		responses = []
		print(len(conversations))
		for conv in conversations:
			for i in range(len(conv) - 1):
				if (re.search(r'__ SILENCE __', conv[i]) or re.search(r'__ SILENCE __', conv[i + 1])):
					continue
				context = re.sub(r'[_]', '', conv[i])
				response = re.sub(r'[_]', '', conv[i + 1])
				contexts.append(context)
				responses.append(response)


		df = pd.read_csv('EmpatheticDialogues.csv', usecols=['conv_id', 'utterance'])
		print('train len' + str(df.shape[0]))
		for i in range(df.shape[0] - 1):
			if (df['conv_id'][i] == df['conv_id'][i + 1]):
				context = re.sub(r'_comma_', ',', df['utterance'][i])
				response = re.sub(r'_comma_', ',', df['utterance'][i + 1])
				contexts.append(context)
				responses.append(response)

		pairs = [(contexts[i], responses[i]) for i in range(len(contexts))]
		random.shuffle(pairs)
		train_x_file = open('train_x2.txt', mode='w+')
		val_x_file = open('val_x.txt2', mode='w+')
		train_y_file = open('train_y.txt2', mode='w+')
		val_y_file = open('val_y.txt2', mode='w+')
		for i in range(int(len(pairs) * (1 - val_rate))):
		    train_x_file.write(pairs[i][0] + '\n')
		    train_y_file.write(pairs[i][1] + '\n')
		for i in range(int(len(contexts) * (1 - val_rate)), len(contexts)):
		    val_x_file.write(pairs[i][0] + '\n')
		    val_y_file.write(pairs[i][1] + '\n')

if __name__ == '__main__':
	main()