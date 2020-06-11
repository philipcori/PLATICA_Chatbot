import json
import re
val_rate = 0.0001
context_length = 3
DELIM = '<delim>'

def main():
	with open('personachat_self_original.json') as json_file:
		data = json.load(json_file)['train']
		conversations = [elt['utterances'][-1]['history'] for elt in data]
		contexts = []
		responses = []
		for conv in conversations:
			for i in range(len(conv)):
				if (i < context_length):
					pass 					# TODO: CREATE DATA FOR FIRST FEW STATEMENTS
				else:
					context = ''
					for j in range(context_length):
						tmpLine = re.sub(r'[_]', '', conv[i - context_length + j])
						context += tmpLine + DELIM
					response = re.sub(r'[_]', '', conv[i])
					if (re.search(r'__ SILENCE __', response) or re.search(r'__ SILENCE __', context)):
						continue
					contexts.append(context)
					responses.append(response)

		train_x_file = open('train_x.txt', mode='w+')
		val_x_file = open('val_x.txt', mode='w+')
		train_y_file = open('train_y.txt', mode='w+')
		val_y_file = open('val_y.txt', mode='w+')
		for i in range(int(len(contexts) * (1 - val_rate))):
		    train_x_file.write(contexts[i] + '\n')
		    train_y_file.write(responses[i] + '\n')
		for i in range(int(len(contexts) * (1 - val_rate)), len(contexts)):
		    val_x_file.write(contexts[i] + '\n')
		    val_y_file.write(responses[i] + '\n')

if __name__ == '__main__':
	main()