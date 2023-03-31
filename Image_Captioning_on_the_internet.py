
import os
import numpy as np 
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from Preprocessing.Preprocessing import *
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Conv2D , MaxPooling2D   , Flatten
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split 
from keras.callbacks import ModelCheckpoint , EarlyStopping
from tensorflow.python.keras.layers.merge import Add
from pickle import load

BASE_DIR = "Flicker_Dataset"
WORKING_DIR  = "Image_Captioning_Project"
directory = os.path.join(BASE_DIR, "Images")

# images = extract_images(directory)
# pickle.dump(images, open(os.path.join(WORKING_DIR, 'images.pkl'), 'wb'))
filename = os.path.join(BASE_DIR, "captions.txt")
doc = load_doc(filename)
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)
vocabulary = to_vocabulary(descriptions)
save_descriptions(descriptions, os.path.join(WORKING_DIR, 'descriptions.txt'))



def Build_CNN_FeatureExtractor(vocab_size, max_length):
      input1 = Input(shape=(214,214,3))
      conv1 = Conv2D(64, (3, 3), activation= tf.nn.relu, padding="same")(input1)
    #   conv2 = Conv2D(64, (3, 3), activation= tf.nn.relu, padding="same")(conv1)
      max1 =  MaxPooling2D((2,2), (2,2))(conv1)
      conv3 = Conv2D(128, (3, 3), activation= tf.nn.relu, padding="same")(max1)
      max2 =  MaxPooling2D((2,2), (2,2))(conv3)
    #   conv4 = Conv2D(128, (3, 3), activation= tf.nn.relu, padding="same")(conv3)
    #   max2 =  MaxPooling2D((2,2), (2,2))(conv4)
    #   conv5 = Conv2D(256, (3, 3), activation= tf.nn.relu, padding="same")(max2)
    #   conv6 = Conv2D(256, (3, 3), activation= tf.nn.relu, padding="same")(conv5)   
    #   max3 =  MaxPooling2D((2,2), (2,2))(conv6)
      flatten = Flatten()(max2)
      dense1 = Dense(1028, activation = tf.nn.relu)(flatten)
      dropout1 = Dropout(0.4)(dense1)
      dense3 = Dense(256)(dropout1)
      input2 = Input(shape = (max_length,))
      embedding = Embedding(input_dim = vocab_size, output_dim = 256)(input2)
      dropout2 = Dropout(0.4)(embedding)
      lstm = LSTM(256)(dropout2)
      added = add([dense3, lstm])
      dense4 = Dense(256 , activation = tf.nn.relu)(added)
      output = Dense(vocab_size, activation = tf.nn.softmax)(dense4)
      model = Model(inputs = [input1 , input2], outputs = output)
      model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics = ["accuracy"])
      return model

filename = os.path.join(BASE_DIR, "Flickr_8k.trainImages.txt")
train = load_set(filename)
train_descriptions = load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), train)
train_images = load_photo_images(os.path.join(WORKING_DIR, "images.pkl"), train)
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max_length(train_descriptions)
filename = os.path.join(BASE_DIR, "Flickr_8k.devImages.txt")
test = load_set(filename)
test_descriptions = load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), test)
test_images = load_photo_images(os.path.join(WORKING_DIR, "images.pkl"), test)
model = Build_CNN_FeatureExtractor(vocab_size, max_length)
model.summary()
epochs = 100
batch_size = 64
steps = len(train) // batch_size 

for i in range(epochs):
  early_stop = EarlyStopping(monitor='val_loss', patience=50)
  generator = create_sequences(tokenizer, max_length, train_descriptions, train_images, vocab_size, batch_size)
  test_generator =  create_sequences(tokenizer, max_length, test_descriptions, test_images, vocab_size, batch_size)
  model.fit(generator, epochs=1, verbose=1, steps_per_epoch = steps , validation_data=(test_generator) , callbacks= [early_stop])