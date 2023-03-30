
import os
import numpy as np 
import pickle
import tensorflow as tf
from tensorflow.keras import models , Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tqdm import tqdm 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Conv2D , MaxPooling2D   , Flatten
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split 
from keras.callbacks import ModelCheckpoint , EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input
from numpy import array
from pickle import load
from tensorflow.python.keras.layers.merge import Add
 
BASE_DIR = "Flicker_Dataset"
WORKING_DIR  = "Image_Captioning_Project"
directory = os.path.join(BASE_DIR, "Images")

# extract features from each photo in the directory
def extract_features(directory):
 # load the model
#  model = VGG16()
 images = {}

 # re-structure the model
#  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
 # summarize
#  print(model.summary())
 # extract features from each photo
#  features = dict()
 for name in tqdm(os.listdir(directory)):
  # load an image from file
  filename = directory + '/' + name
  image = load_img(filename, target_size=(28, 28))
  # convert the image pixels to a numpy array
  image = img_to_array(image)
  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  # prepare the image for the VGG model
  image = preprocess_input(image)
  # get features
  # feature = model.predict(image, verbose=0)
  # get image id
  image_id = name.split('.')[0]
  images[image_id] = image

  # store feature
  # features[image_id] = feature
 return images
 
# extract features from all images
images = extract_features(directory)
print('Extracted images: %d' % len(images))
# save to file
pickle.dump(images, open(os.path.join(WORKING_DIR, 'images.pkl'), 'wb'))

import string

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
filename = os.path.join(BASE_DIR, "captions.txt")

# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, os.path.join(WORKING_DIR, 'descriptions.txt'))


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# load photo images
def load_photo_images(filename, dataset):
	# load all images
	all_images = load(open(filename, 'rb'))
	# filter features
	images = {k: all_images[k] for k in dataset}
	return images

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
  #    if key != "":
  #           np.append(X1,(photos[key][0]))
  #           np.append(X2, in_seq)
  #           np.append(y, out_seq)
  # # X1, X2, y = np.empty(), np.empty(), np.empty()

  # return X1,X2,y
  X1, X2, y = list(), list(), list()
  # walk through each image identifier
  for key, desc_list in descriptions.items():
    # walk through each description for the image
    for desc in desc_list:
      # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
         # split into input and output pair
          in_seq, out_seq = seq[:i], seq[i]
          # pad input sequence
          in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          # encode output sequence
          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          # store
          if key != "":
            X1.append((photos[key][0]))
            X2.append(in_seq)
            y.append(out_seq)
  return np.array(X1), np.array(X2), np.array(y)


def Build_CNN_FeatureExtractor(vocab_size, max_length):
      input1 = Input(shape=(28,28,3))
      conv1 = Conv2D(64, (3, 3), activation= tf.nn.relu, padding="same")(input1)
      conv2 = Conv2D(64, (3, 3), activation= tf.nn.relu, padding="same")(conv1)
      max1 =  MaxPooling2D((2,2), (2,2))(conv2)
      conv3 = Conv2D(128, (3, 3), activation= tf.nn.relu, padding="same")(max1)
      conv4 = Conv2D(128, (3, 3), activation= tf.nn.relu, padding="same")(conv3)
      max2 =  MaxPooling2D((2,2), (2,2))(conv4)
      conv5 = Conv2D(256, (3, 3), activation= tf.nn.relu, padding="same")(max2)
      conv6 = Conv2D(256, (3, 3), activation= tf.nn.relu, padding="same")(conv5)
      conv7 = Conv2D(256, (3, 3), activation= tf.nn.relu, padding="same")(conv6)
      max3 =  MaxPooling2D((2,2), (2,2))(conv7)
      conv8 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(max3)
      conv9 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(conv8)
      conv10 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(conv9)
      max4 =  MaxPooling2D((2,2), (2,2))(conv10)
    #   conv11 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(max4)
    #   conv12 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(conv11)
    #   conv13 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(conv12)
    #   max5 =  MaxPooling2D((2,2), (2,2))(conv13)
      flatten = Flatten()(max4)
      dense1 = Dense(4096, activation = tf.nn.relu)(flatten)
      dropout1 = Dropout(0.4)(dense1)
      dense2 = Dense(4096)(dropout1)
      dense3 = Dense(256)(dense2)
      input2 = Input(shape = (max_length,))
      embedding = Embedding(input_dim = vocab_size, output_dim = 256)(input2)
      dropout2 = Dropout(0.4)(embedding)
      lstm = LSTM(256)(dropout2)
      added = add([dense3, lstm])
      dense4 = Dense(256 , activation = tf.nn.relu)(added)
      output = Dense(vocab_size, activation = tf.nn.softmax)(dense4)
      model = Model(inputs = [input1 , input2], outputs = output)
      model.compile(loss='categorical_crossentropy', optimizer='adam')
      return model
# train dataset

# load training dataset (6K)
filename = os.path.join(BASE_DIR, "Flickr_8k.trainImages.txt")
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_images = load_photo_images(os.path.join(WORKING_DIR, 'images.pkl'), train)
print('Photos: train=%d' % len(train_images))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_images, vocab_size)


# dev dataset

# load test set
filename = os.path.join(BASE_DIR, "Flickr_8k.devImages.txt")
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_images = load_photo_images(os.path.join(WORKING_DIR, 'images.pkl'), test)
print('Photos: test=%d' % len(test_images))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_images, vocab_size)



# fit model

# define the model
model = Build_CNN_FeatureExtractor(vocab_size, max_length)
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
# earlystopping = Ear
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, batch_size = 128,  verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))