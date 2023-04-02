
import os
import time
import numpy as np 
import pickle
from Preprocessing.Preprocessing import *
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split 
from keras.callbacks import ModelCheckpoint , EarlyStopping
from tensorflow.python.keras.layers.merge import Add
from pickle import load
from Neural_Networks.Custum_CNN_LSTM import Custum_CNN_LSTM
BASE_DIR = "Flicker_Dataset"
WORKING_DIR  = "Image_Captioning_Project"
directory = os.path.join(BASE_DIR, "Images")
Read = True 

# Preprocessing 
if Read == True :
	images = extract_images(directory)
	pickle.dump(images, open(os.path.join(WORKING_DIR, 'images.pkl'), 'wb'))
filename = os.path.join(BASE_DIR, "captions.txt")
doc = load_doc(filename)
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)
vocabulary = to_vocabulary(descriptions)
save_descriptions(descriptions, os.path.join(WORKING_DIR, 'descriptions.txt'))
filename = os.path.join(BASE_DIR, "Flickr_8k.trainImages.txt")
train = load_set(filename)
train_descriptions = load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), train)
test = load_set(filename)
train_images = load_photo_images(Read, os.path.join(WORKING_DIR, "images.pkl"), train, images)
test_images = load_photo_images(Read, os.path.join(WORKING_DIR, "images.pkl"), test, images)
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max_length(train_descriptions)
filename = os.path.join(BASE_DIR, "Flickr_8k.devImages.txt")
test_descriptions = load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), test)
image = test_images[(list(test_images.keys())[0])].shape
image_size= (image[1], image[2], image[3])

# Loading the custum network
Custum_neural_network = Custum_CNN_LSTM(vocab_size , image_size , max_length)

model = Custum_neural_network.Build_CNN_FeatureExtractor()
model.summary()
epochs = 100
batch_size = 64
steps = len(train) // batch_size 


# Running the model
for i in range(epochs):
  early_stop = EarlyStopping(monitor='val_loss', patience=5)
  generator = create_sequences(tokenizer, max_length, train_descriptions, train_images, vocab_size, batch_size)
  test_generator =  create_sequences(tokenizer, max_length, test_descriptions, test_images, vocab_size, batch_size)
  model.fit(generator, epochs=1, verbose=1, steps_per_epoch = steps , validation_data=(test_generator) , callbacks= [early_stop])