import json
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

		pairs = [(contexts[i], responses[i]) for i in range(len(contexts))]
		random.shuffle(pairs)
		train_x_file = open('train_x.txt', mode='w+')
		val_x_file = open('val_x.txt', mode='w+')
		train_y_file = open('train_y.txt', mode='w+')
		val_y_file = open('val_y.txt', mode='w+')
		for i in range(int(len(pairs) * (1 - val_rate))):
		    train_x_file.write(pairs[i][0] + '\n')
		    train_y_file.write(pairs[i][1] + '\n')
		for i in range(int(len(contexts) * (1 - val_rate)), len(contexts)):
		    val_x_file.write(pairs[i][0] + '\n')
		    val_y_file.write(pairs[i][1] + '\n')

if __name__ == '__main__':
	main()