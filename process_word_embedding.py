from gensim.models.word2vec import Word2Vec
import numpy as np

model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# model = Word2Vec.load('GoogleNews-vectors-negative300.bin')
wordVocab = [k for (k, v) in model.vocab.iteritems()]

index_dict = {}
word_vectors = {}
counter = 1
for word in wordVocab:
	index_dict[word] = counter
	counter += 1
	word_vectors[word] = model[word]
	print word_vectors[word]
	break

vocab_dim = 300 # dimensionality of your word vectors
n_symbols = len(index_dict) + 1 # adding 1 to account for 0th index (for masking)
embedding_weights = np.zeros((n_symbols+1,vocab_dim))
for word,index in index_dict.items():
    embedding_weights[index,:] = word_vectors[word]



# assemble the model
model = Sequential() # or Graph or whatever
model.add(Embedding(output_dim=rnn_dim, input_dim=n_symbols + 1, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
model.add(LSTM(dense_dim, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(n_symbols, activation='softmax')) # for this is the architecture for predicting the next word, but insert your own here
