import json
import re

def main():
	with open('personachat_self_original.json') as json_file:
		data = json.load(json_file)['train']
		conversations = [elt['utterances'][-1]['history'] for elt in data]
		word_counts = dict()
		stmt_counts = dict()
		for conv in conversations:
			for stmt in conv:
				if stmt in stmt_counts:
					stmt_counts[stmt] += 1
				else:
					stmt_counts[stmt] = 1
				for word in stmt.split(' '):
					if word in word_counts:
						word_counts[word] += 1
					else:
						word_counts[word] = 1
		print('**********WORD COUNTS*************')
		i = 0
		for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True):
			print(k, v)
			i += 1
			if (i > 20):
				break
		i = 0
		print('vocab size:' + str(len(word_counts)))
		print('**********STATEMENT COUNTS********')
		for k, v in sorted(stmt_counts.items(), key=lambda item: item[1], reverse=True):
			print(k, v)
			i += 1
			if (i > 20):
				break		
		# print(stmt_counts)


if __name__ == '__main__':
	main()