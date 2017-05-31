'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from gensim.models.word2vec import Word2Vec
import numpy as np
import gensim



# model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# model = Word2Vec.load('GoogleNews-vectors-negative300.bin')
wordVocab = [k for (k, v) in model.vocab.iteritems()]

index_dict = {}
word_vectors = {}
counter = 1
for word in wordVocab:
	index_dict[word] = counter
	counter += 1
	word_vectors[word] = model[word]

vocab_dim = 300 # dimensionality of your word vectors
n_symbols = len(index_dict) + 1 # adding 1 to account for 0th index (for masking)
embedding_weights = np.zeros((n_symbols+1,vocab_dim))
for word,index in index_dict.items():
	embedding_weights[index,:] = word_vectors[word]



# -----------------------------------------------------





max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
(x_train, y_train), (x_test, y_test) = imdb.load_data()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()


# -----------------------------------------------------

# model.add(Embedding(max_features, 128))
# model.add(Embedding(output_dim=max_features, input_dim=n_symbols + 1, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
model.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols + 1, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention

# -----------------------------------------------------

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=15,
		  validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
							batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

