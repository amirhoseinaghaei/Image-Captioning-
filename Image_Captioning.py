import itertools
import os
import time
import numpy as np 
import pickle
from Neural_Networks.VGG16_Custum_CNN import VGG16_Custum_CNN
from Preprocessing.Preprocessing import *
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split 
from keras.callbacks import ModelCheckpoint , EarlyStopping
from tensorflow.python.keras.layers.merge import Add
from pickle import load
from Neural_Networks.Custum_CNN_LSTM import Custum_CNN_LSTM
from Preprocessing.Preprocessing import preprocessing
BASE_DIR = "Flicker_Dataset"
WORKING_DIR  = "Image_Captioning_Project"
directory = os.path.join(BASE_DIR, "Images")
Read = False
Custum = False
# loading preprocessor
preprocessor = preprocessing(directory , 10)
images = 0
features = 0
# Preprocessing 
if Read == True :
  if ~Custum:
    features = preprocessor.extract_features()
    pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
  else:
   images = preprocessor.extract_images() 
   pickle.dump(images, open(os.path.join(WORKING_DIR, 'images.pkl'), 'wb'))
filename = os.path.join(BASE_DIR, "captions.txt")
doc = preprocessor.load_doc(filename)
doc = preprocessor.load_doc(filename)
descriptions = preprocessor.load_descriptions(doc)
preprocessor.clean_descriptions(descriptions)
vocabulary = preprocessor.to_vocabulary(descriptions)
preprocessor.save_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), descriptions)
filename = os.path.join(BASE_DIR, "Flickr_8k.trainImages.txt")
train = preprocessor.load_set(filename)
train_descriptions = preprocessor.load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), train)
tokenizer = preprocessor.create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
max_length = preprocessor.max_length(train_descriptions)
filename = os.path.join(BASE_DIR, "Flickr_8k.devImages.txt")
test = preprocessor.load_set(filename)
test_descriptions = preprocessor.load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), test)
filename = os.path.join(BASE_DIR, "Flickr_8k.testImages.txt")
Rtest = preprocessor.load_set(filename)
R_test_descriptions = preprocessor.load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), Rtest)
train_descriptions.update(R_test_descriptions)
if Custum:
  train_images = preprocessor.load_photo_images(os.path.join(WORKING_DIR, "images.pkl"),Read, train, images)
  test_images = preprocessor.load_photo_images(os.path.join(WORKING_DIR, "images.pkl"), Read, test, images)
  image = test_images[(list(test_images.keys())[0])].shape
  image_size= (image[1], image[2], image[3])
  Custum_neural_network = Custum_CNN_LSTM(vocab_size , image_size , max_length)
  # Building the model for custum CNN-LSTM model 
  model1 = Custum_neural_network.Build_CNN_FeatureExtractor()
  model1.summary()


else:
  train_features = preprocessor.load_photo_features(os.path.join(WORKING_DIR, 'features.pkl'), Read, train, features)
  test_features = preprocessor.load_photo_features(os.path.join(WORKING_DIR, 'features.pkl'), Read, test, features)
  Pretrained_neural_network = VGG16_Custum_CNN(vocab_size , max_length)
  # Building the model for custum VGG16-Custm LSTM model 
  Rtest_features = preprocessor.load_photo_features(os.path.join(WORKING_DIR, 'features.pkl'), Read, Rtest, features)
  train_features.update(Rtest_features)
  # Building the model for custum VGG16-Custm LSTM model 
  model2 = Pretrained_neural_network.define_model()
  model2.summary()





epochs = 100
batch_size = 64
steps = len(train) // batch_size 

path = "model" + f"{2 if ~Custum else 1}" + "-val_loss:_{val_loss:.3f}.h5"
filepath = os.path.join(WORKING_DIR, path)
model = model1 if Custum else model2
# Running the model
for i in range(epochs):
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  early_stop = EarlyStopping(monitor='val_loss', patience=5)
  generator = preprocessor.create_sequences(tokenizer, max_length, train_descriptions, train_images if Custum else train_features, vocab_size, batch_size)
  test_generator =  preprocessor.create_sequences(tokenizer, max_length, test_descriptions, test_images if Custum else test_features, vocab_size, batch_size)
  model.fit(generator, epochs=1, verbose=1, steps_per_epoch = steps , validation_data=(test_generator) , callbacks= [checkpoint, early_stop])
