# LOADING LIBRARIES

import pandas as pd                 # for working with data
import numpy as np                  # for scientific computing
import re                           # for regular expressions
import nltk                         # for text manipulation
from nltk.corpus import stopwords   # for removing stopwords in English
import matplotlib.pyplot as plt     # To plot graphs


from numpy import array
from keras.preprocessing.text import one_hot                                                # for performing one hot encoding
from keras.preprocessing.sequence import pad_sequences                                      # To make all sequences of same length
from keras.models import Sequential                                                         # Importing sequential model
from keras.layers.recurrent import LSTM                                                     # Importing LSTM model
from keras.layers import Activation, Dropout, Dense, Flatten, Conv1D, GlobalMaxPooling1D    # To reshape the layers of the model
from keras.layers.embeddings import Embedding                                               # Converting words into low dimension vectors
from sklearn.model_selection import train_test_split                                        # For splitting data into train and test data
from keras.preprocessing.text import Tokenizer # To split documents into tokens
from keras.preprocessing.text import text_to_word_sequence # To convert words into integers

np.random.seed(11)      # To reproduce results

###################################################################################################################################################################

# DATA INSPECTION

# Reading dataset
movie_reviews = pd.read_csv("IMDB-Dataset.csv")
# Checking for any null entries in dataset
print("\nIs the dataset having any null entries?")
print(movie_reviews.isnull().values.any())
# Printing dataset dimensions
print("\nDataset dimensions")
print(movie_reviews.shape)
# First 5 entries in datatset
print("\nFirst 5 entries in datatset")
print(movie_reviews.head())
# Number of positive and negative reviews in dataset
print("\nNumber of positive and negative reviews in dataset")
print(movie_reviews['sentiment'].value_counts())

###################################################################################################################################################################

# DATA PREPROCESSING

# Sample review
print("\nSAMPLE REVIEW")
print(movie_reviews['review'][4])



#Function to preprocess each review
    # 'r' is used to differentiate regex escaping and normal escape sequence 
def preprocess(sen):
    # Function to remove html tags is called
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    # '^' means NOT
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

# Function to remove html tags
def remove_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# Passing all the reviews into preprocess function defined earlier
    #Defining an empty list to store preprocessed text
X = []
# Converting our dataset from type "dataframe" to type "list"
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess(sen))



# Sample review after preprocessing
print("\nSAMPLE REVIEW AFTER PREPROCESSING")
print(X[4])

# One hot encoding of target feature
y = movie_reviews['sentiment']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

#Splitting dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=70)

###################################################################################################################################################################

# DATA PREPARATION

# Consider only top 5000 most frequent words
    # Converts words into vectors holding sequence of intergers for corresponding words
tokenizer = Tokenizer(num_words=5000)
# Generate index for each string based on frequency (more frequent ; lesser the index value)
tokenizer.fit_on_texts(X_train)
print("\nNumber of unique words in training set") 
print(len(tokenizer.word_index))
# Substitute words with corresponding word index values
  # Only the words available in "fit_on_texts" will be considered. Others will be ignored  
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)



# Adding '1' as index '0' is reserved in word_index
vocab_size = len(tokenizer.word_index) + 1
# Setting maximum length of each list(review) to be of length 100 words
maxlen = 100
# Making the train and test statements to be of size 100 by truncating or padding accordingly
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



# Creating a dictionary with words as key and corresponding embedding list loaded from "GloVe" dataset as values
embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")
# Creating each line in GloVe dataset as a kwy-value pair
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()



# Creating embedding matrix
embedding_matrix = np.zeros((vocab_size, 100))
# Getting vector representation (from embedding dictionary) of each word in word_index
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    #If word is not available in GloVE embedding text file, that word will be skipped
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

###################################################################################################################################################################

# CONSTRUCTING SIMPLE DEEP NEURAL NETWORK

# Creating a sequential model
dnn_model = Sequential()
# Setting imput and output size to 100. Since we are using "GloVe - a predefined embedding", we set "trainable" to False
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
dnn_model.add(embedding_layer)
# Flattening the embedding the layer
    # layer of dimension (5, 3) becomes (1, 15)
dnn_model.add(Flatten())
# Adding Dense layer with one dimension output space
    # Dense is made at final layer - Fully connected layer
dnn_model.add(Dense(1, activation='sigmoid'))

# Compiling the model
    #Optimizer used for faster convergence of model training
dnn_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])
# Printing the model summary
print("\nSIMPLE DEEP NEURAL NETWORK MODEL")
print(dnn_model.summary())

# Training our model
dnn_history = dnn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
# Evaluating the model's accuracy
dnn_score = dnn_model.evaluate(X_test, y_test, verbose=1)
print("\nTEST SCORES OF MODEL")
print("Test loss:", dnn_score[0])
print("Test Accuracy:", dnn_score[1])

# Plotting graph for model built
    # Accuracy graph
plt.title('Neural network model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epoch')
plt.plot(dnn_history.history['acc'], label = 'Train')
plt.plot(dnn_history.history['val_acc'], label = 'Valid')
plt.legend(loc='best')
plt.show()
    # Loss graph
plt.title('Neural network model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(dnn_history.history['loss'], label = 'Train')
plt.plot(dnn_history.history['val_loss'], label = 'Valid')
plt.legend(loc='best')
plt.show()

###################################################################################################################################################################

# CONSTRUCTING CONVOLUTION NEURAL NETWORK

# Creating a sequential model
cnn_model = Sequential()
# Setting imput and output size to 100. Since we are using "GloVe - a predefined embedding", we set "trainable" to False
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
cnn_model.add(embedding_layer)
# We use 1D CNN as 2D CNN is genrally used for images
    # 128 filters used - 128 different features can be detected
    # Window size is 5 - will consider 5 entries at a time
cnn_model.add(Conv1D(128, 5, activation='relu'))
# Pooling is done to avoid overfitting - also reduces the dimension of data
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(1, activation='sigmoid'))

#Compiling the model
cnn_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])

# Printing the model summary
print("\nCONVOLUTIONAL NEURAL NETWORK 1D MODEL")
print(cnn_model.summary())

# Training our model
cnn_history = cnn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
# Evaluating the model's accuracy
cnn_score = cnn_model.evaluate(X_test, y_test, verbose=1)
print("\nTEST SCORES OF MODEL")
print("Test loss:", cnn_score[0])
print("Test Accuracy:", cnn_score[1])

# Plotting graph for model built
    # Accuracy graph
plt.title('CNN model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epoch')
plt.plot(cnn_history.history['acc'], label = 'Train')
plt.plot(cnn_history.history['val_acc'], label = 'Valid')
plt.legend(loc='best')
plt.show()
    # Loss graph
plt.title('CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(cnn_history.history['loss'], label = 'Train')
plt.plot(cnn_history.history['val_loss'], label = 'Valid')
plt.legend(loc='best')
plt.show()

###################################################################################################################################################################

# LSTM RECURRENT NEURAL NETWORK

# Creating a sequential model
rnn_model = Sequential()
# Setting imput and output size to 100. Since we are using "GloVe - a predefined embedding", we set "trainable" to False
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
rnn_model.add(embedding_layer)
# LSTM model is created with 128 neurons
rnn_model.add(LSTM(128))
rnn_model.add(Dense(1, activation='sigmoid'))

#Compiling the model
rnn_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])

# Printing the model summary
print("\nLSTM RECURRENT NEURAL NETWORK MODEL")
print(rnn_model.summary())

# Training our model
rnn_history = rnn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
# Evaluating the model's accuracy
rnn_score = rnn_model.evaluate(X_test, y_test, verbose=1)
print("\nTEST SCORES OF MODEL")
print("Test loss:", rnn_score[0])
print("Test Accuracy:", rnn_score[1])

# Plotting graph for model built
    # Accuracy graph
plt.title('LSTM model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epoch')
plt.plot(rnn_history.history['acc'], label = 'Train')
plt.plot(rnn_history.history['val_acc'], label = 'Valid')
plt.legend(loc='best')
plt.show()
    # Loss graph
plt.title('LSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(rnn_history.history['loss'], label = 'Train')
plt.plot(rnn_history.history['val_loss'], label = 'Valid')
plt.legend(loc='best')
plt.show()

###################################################################################################################################################################

# Final Results
print("\nDeep Neural Network")
print("Traning Accuracy : {}" .format(dnn_history.history['acc']))
print("Validation Accuracy : {}" .format(dnn_history.history['val_acc']))
print("\nTraning Loss : {}" .format(dnn_history.history['loss']))
print("Validation Loss : {}" .format(dnn_history.history['val_loss']))

print("\n\n\nConvolution Neural Network")
print("Traning Accuracy : {}" .format(cnn_history.history['acc']))
print("Validation Accuracy : {}" .format(cnn_history.history['val_acc']))
print("\nTraning Loss : {}" .format(cnn_history.history['loss']))
print("Validation Loss : {}" .format(cnn_history.history['val_loss']))

print("\n\n\nLSTM")
print("Traning Accuracy : {}" .format(rnn_history.history['acc']))
print("Validation Accuracy : {}" .format(rnn_history.history['val_acc']))
print("\nTraning Loss : {}" .format(rnn_history.history['loss']))
print("Validation Loss : {}" .format(rnn_history.history['val_loss']))

###################################################################################################################################################################      

# Making prediction on any single review using three models that we have built

# Getting input from user
print("\nMAKING PREDICTION ON SINGLE INSTANCE USING THREE MODELS THAT WE HAVE BUILT")
reviewNo = int(input("\nEnter the review number whose sentiment is to be predcited"))
reviewText = X[reviewNo]
print("\nCONTENT OF THE REVIEW CHOSEN IS")
print(reviewText)

# Converting text to numeric form
    #Using the tokenizer built earlier
#Since we have trained with a list of reviews and now we are feeding in a string, we need to apply "text_to_word_sequence" before tokenizing
reviewText = text_to_word_sequence(reviewText)
reviewProcessed = tokenizer.texts_to_sequences(reviewText)
print("\nINTERGER SEQUENCE OF THE REVIEW CHOSEN IS")
print(reviewProcessed)

flat_list = []
for sublist in reviewProcessed:
    for item in sublist:
        flat_list.append(item)
#Making the entire items as a single list
flat_list = [flat_list]
#Padding to fit into model
reviewSequence = pad_sequences(flat_list, padding='post', maxlen=maxlen)


#Predicting the sentiment of review using three models
print("\n'0 to 0.5' - Negative ; '0.51 - 1' - Positive")
print("\nPredition by Simple Neural network model is")
print(dnn_model.predict(reviewSequence))
print("\nPredition by Convolution Neural network model is")
print(cnn_model.predict(reviewSequence))
print("\nPredition by LSTM Recurrent Neural network model is")
print(rnn_model.predict(reviewSequence))
