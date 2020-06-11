import pickle
import numpy as np

f = open('glove.6B.100d.txt', 'r')
g = open('glove.6B.100d_pickle', 'wb')
word_dict = {}
wordvec = []
for idx, line in enumerate(f.readlines()):
    word_split = line.split(' ')
    word = word_split[0]
    word_dict[word] = idx
    d = word_split[1:]
    d[-1] = d[-1][:-1]
    d = [float(e) for e in d]
    wordvec.append(d)

embedding = np.array(wordvec)
pickling = {}
pickling = {'embedding' : embedding, 'word_dict': word_dict}
pickle.dump(pickling, g)
f.close()
g.close()