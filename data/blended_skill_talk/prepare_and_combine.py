import json
import random
	
val_rate = 0.001
contexts = []
responses = []

with open('train.json') as file:
	contents = json.load(file)
	for conv in contents:
		dialog = conv['dialog']
		for i in range(len(dialog) - 1):
			context = dialog[i][1]
			response = dialog[i + 1][1]
			contexts.append(context)
			responses.append(response)

with open('test.json') as file:
	contents = json.load(file)
	for conv in contents:
		dialog = conv['dialog']
		for i in range(len(dialog) - 1):
			context = dialog[i][1]
			response = dialog[i + 1][1]
			contexts.append(context)
			responses.append(response)

with open('valid.json') as file:
	contents = json.load(file)
	for conv in contents:
		dialog = conv['dialog']
		for i in range(len(dialog) - 1):
			context = dialog[i][1]
			response = dialog[i + 1][1]
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