
import itertools
import os
import time
import numpy as np 
import pickle
from Neural_Networks.PretrainedModels import PretrainedModels
from Preprocessing.Preprocessing import *
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from sklearn.model_selection import train_test_split 
from keras.callbacks import ModelCheckpoint , EarlyStopping
from tensorflow.python.keras.layers.merge import Add
from pickle import load
import tensorflow as tf
from Neural_Networks.Custum_CNN_LSTM import Custum_CNN_LSTM
from Preprocessing.Preprocessing import preprocessing
from nltk.translate.bleu_score import corpus_bleu
# from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input


BASE_DIR = "Flicker_Dataset"
WORKING_DIR  = "Image_Captioning_Project"
RealImages_directory = os.path.join(BASE_DIR, "Real_Images")  
directory = os.path.join(BASE_DIR, "Images")
Read = False 
Custum = True
# loading preprocessor
preprocessor = preprocessing(directory , 32)
images = 0
features = 0
# Preprocessing 
if Read == True :
  if Custum == False:
    features = preprocessor.extract_features()
    pickle.dump(features, open(os.path.join(WORKING_DIR, 'featuresResNet.pkl'), 'wb'))
  else:
   images = preprocessor.extract_images() 
   pickle.dump(images, open(os.path.join(WORKING_DIR, 'images.pkl'), 'wb'))
filename = os.path.join(BASE_DIR, "captions.txt")
doc = preprocessor.load_doc(filename)
descriptions = preprocessor.load_descriptions(doc)


doc30k = preprocessor.load_doc1(os.path.join(BASE_DIR, "caption30k.txt"))
descriptions30k = preprocessor.load_descriptions(doc30k)
preprocessor.clean_descriptions(descriptions)
preprocessor.clean_descriptions(descriptions30k)
# descriptions.update(descriptions30k)
vocabulary = preprocessor.to_vocabulary(descriptions)
preprocessor.save_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), descriptions)
preprocessor.save_descriptions(os.path.join(WORKING_DIR, 'descriptions30k.txt'), descriptions30k)

filename = os.path.join(BASE_DIR, "Flickr_8k.trainImages.txt")
train = preprocessor.load_set(filename)
train_descriptions = preprocessor.load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions.txt'), train)
filename = os.path.join(BASE_DIR, "Flicker30k_train.txt")
train_30k = preprocessor.load_set(filename)
train_descriptions30k = preprocessor.load_clean_descriptions(os.path.join(WORKING_DIR, 'descriptions30k.txt'), train_30k)

tokenizer = preprocessor.create_tokenizer(train_descriptions)

# train_descriptions.update(train_descriptions30k)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
max_length = preprocessor.max_length(train_descriptions)
# max_length = 37
# print(max_length)
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
  # model1 = Custum_neural_network.Build_CNN_FeatureExtractor()
  # model1.summary()

  modelCustom = tf.keras.models.load_model(os.path.join(WORKING_DIR, 'modelCustom2-val_loss_3.290.h5'))

else:
  # train_features = preprocessor.load_photo_features(os.path.join(WORKING_DIR, 'featuresMobileNetk.pkl'), Read, train, features)
  # train_features30k = preprocessor.load_photo_features(os.path.join(WORKING_DIR, 'featuresMobileNetk.pkl'), Read, train_30k, features)
  # crowler_dict = {}
  # for key in tqdm(train_features30k.keys()):
  #   if key in train_descriptions30k.keys():
  #     crowler_dict[key] = train_features30k[key]
  # train_features30k = crowler_dict
  test_features = preprocessor.load_photo_features(os.path.join(WORKING_DIR, 'featuresResNet.pkl'), Read, test, features)
  Pretrained_neural_network = PretrainedModels(vocab_size , max_length)
  # Building the model for custum VGG16-Custm LSTM model 
  Rtest_features = preprocessor.load_photo_features(os.path.join(WORKING_DIR, 'featuresResNet.pkl'), Read, Rtest, features)
  # train_features.update(train_features30k)
  # # model2 = Pretrained_neural_network.define_model2()
  # model30k_VGG16 = tf.keras.models.load_model(os.path.join(WORKING_DIR, 'model30k2-val_loss_3.319.h5'))
  # model8K_VGG16 = tf.keras.models.load_model(os.path.join(WORKING_DIR, 'model2VGG16-val_loss__3.276.h5'))
  model8K_Resnet50 = tf.keras.models.load_model(os.path.join(WORKING_DIR, 'model2Resnet50-val_loss__3.210.h5'))
  # model8K_MobileNetV2 = tf.keras.models.load_model(os.path.join(WORKING_DIR, 'modelMobileNet2-val_loss__3.147.h5'))

  # print(len(train_descriptions30k))

  # print(len(train_features30k))
  # model2.summary()





# # epochs = 20
# # batch_size = 64
# # steps = len(train) // batch_size 

# # path = "model" + f"{2 if ~Custum else 1}" + "-val_loss:_{val_loss:.3f}.h5"
# # filepath = os.path.join(WORKING_DIR, path)
# # model = model1 if Custum else model2
# # # Running the model
# # for i in range(epochs):
# #   checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# #   early_stop = EarlyStopping(monitor='val_loss', patience=5)
# #   generator = preprocessor.create_sequences(tokenizer, max_length, train_descriptions, train_images if Custum else train_features, vocab_size, batch_size)
# #   test_generator =  preprocessor.create_sequences(tokenizer, max_length, test_descriptions, test_images if Custum else test_features, vocab_size, batch_size)
# #   model.fit(generator, epochs=1, verbose=1, steps_per_epoch = steps , validation_data=(test_generator) , callbacks= [checkpoint, early_stop])




def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text
from nltk.translate.bleu_score import corpus_bleu


def extract_features(directory):
  # model = VGG16()
  # model = MobileNetV2()
  
  model = ResNet50()
  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
  features = dict()
  for name in tqdm(os.listdir(directory)):
    filename = directory + '/' + name
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = name.split('.')[0]
    features[image_id] = feature
  return features

features = extract_features(RealImages_directory)
print(features)
def extract_images(directory):
  images = {}
  for name in tqdm(os.listdir(directory)):
    filename = directory + '/' + name
    image = load_img(filename, target_size=(32, 32))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    image_id = name.split('.')[0]
    images[image_id] = image
  return images
real_iamges = extract_images(RealImages_directory)
print(real_iamges)
actual, y_pred_30K_VGG16_predicted , y_pred_8K_VGG16_predicted , y_pred_8K_Resnet50_predicted =  list(), list() , list() , list()
y_pred_8K_MobileNetV2_predicted = list()
y_pred_Custom_Predicted = list()
i = 0
item = 920
name = ""
for key in tqdm(test_descriptions.keys()):
    # get actual caption
    if i == item:
      name = key

      captions = descriptions[key]
      # predict the caption for image
      # y_pred_30K_VGG16 = predict_caption(model30k_VGG16, test_features[key], tokenizer, max_length) 
      # y_pred_8K_VGG16 = predict_caption(model8K_VGG16, test_features[key], tokenizer, max_length) 
      # y_pred_Custom = predict_caption(modelCustom, test_images[key], tokenizer, max_length) 
      # y_pred_8K_Resnet50 = predict_caption(model8K_Resnet50, test_features[key], tokenizer, max_length) 
      # y_pred_8K_MobileNetV2 = predict_caption(model8K_MobileNetV2, test_features[key], tokenizer, max_length) 
      y_pred_Custom = predict_caption(modelCustom, test_images[key], tokenizer, max_length) 

      # split into words
      y_pred_Custom = y_pred_Custom.split()
      # y_pred_8K_VGG16 = y_pred_8K_VGG16.split()
      # y_pred_30K_VGG16 = y_pred_30K_VGG16.split()
      # y_pred_8K_MobileNetV2 = y_pred_8K_MobileNetV2.split()
      # y_pred_8K_Resnet50 = y_pred_8K_Resnet50.split()

      actual_captions = [caption.split() for caption in captions]
      # append to the list

      actual.append(actual_captions)
      # y_pred_30K_VGG16_predicted.append(y_pred_30K_VGG16)
      # y_pred_8K_VGG16_predicted.append(y_pred_8K_VGG16)
      # y_pred_8K_Resnet50_predicted.append(y_pred_8K_Resnet50)
      # print("BLEU-1 for y_pred_8K_Resnet50_predicted: %f" % corpus_bleu(actual, y_pred_8K_Resnet50_predicted, weights=(1.0, 0, 0, 0)))

      y_pred_Custom_Predicted.append(y_pred_Custom)
      # if corpus_bleu(actual, y_pred_30K_VGG16_predicted, weights=(1.0, 0, 0, 0)) < 0.3:
      #    print(i)
      #    item = i
      #    name = key
      #    break
      # print(item)
      # y_pred_30K_VGG16_predicted , actual = list(), list()
      # y_pred_8K_MobileNetV2_predicted.append(y_pred_8K_MobileNetV2)
      #    item = i
      #    name = key
    i += 1

    
# calcuate BLEU score
# print("BLEU-1 for y_pred_30K_VGG16_predicted: %f" % corpus_bleu(actual, y_pred_30K_VGG16_predicted, weights=(1.0, 0, 0, 0)))
# print("BLEU-2 for y_pred_30K_VGG16_predicted: %f" % corpus_bleu(actual, y_pred_30K_VGG16_predicted, weights=(0.5, 0.5, 0, 0)))
# print("BLEU-1 for y_pred_8K_VGG16_predicted: %f" % corpus_bleu(actual, y_pred_8K_VGG16_predicted, weights=(1.0, 0, 0, 0)))
# print("BLEU-2 for y_pred_8K_VGG16_predicted: %f" % corpus_bleu(actual, y_pred_8K_VGG16_predicted, weights=(0.5, 0.5, 0, 0)))
# print("BLEU-1 for y_pred_8K_Resnet50_predicted: %f" % corpus_bleu(actual, y_pred_8K_Resnet50_predicted, weights=(1.0, 0, 0, 0)))
# print("BLEU-2 for y_pred_8K_Resnet50_predicted: %f" % corpus_bleu(actual, y_pred_8K_Resnet50_predicted, weights=(0.5, 0.5, 0, 0)))
# print("BLEU-1 for y_pred_8K_VGG16_predicted: %f" % corpus_bleu(actual, y_pred_8K_MobileNetV2_predicted, weights=(1.0, 0, 0, 0)))
# print("BLEU-2 for y_pred_8K_VGG16_predicted: %f" % corpus_bleu(actual, y_pred_8K_MobileNetV2_predicted, weights=(0.5, 0.5, 0, 0)))
print("BLEU-1 for custom: %f" % corpus_bleu(actual, y_pred_Custom_Predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2 for custom: %f" % corpus_bleu(actual, y_pred_Custom_Predicted, weights=(0.5, 0.5, 0, 0)))
from PIL import Image
import matplotlib.pyplot as plt
image_name = f"IMG_2185.jpg"
def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Real_Images", image_name)
    image = Image.open(img_path)
    # captions = descriptions[image_id]
    # print('---------------------Actual---------------------')
    # for caption in captions:
    #     print(caption)
    # y_pred_30K_VGG16 = predict_caption(model30k_VGG16, features[image_id], tokenizer, max_length) 
    # y_pred_8K_VGG16 = predict_caption(model8K_VGG16, features[image_id], tokenizer, max_length) 
    # y_pred_8K_MobileNetV2 = predict_caption(model8K_MobileNetV2, features[image_id], tokenizer, max_length) 
    # y_pred_8K_MobileNetV2 = predict_caption(model8K_Resnet50, features[image_id], tokenizer, max_length) 

    y_pred_Custom = predict_caption(modelCustom, real_iamges[image_id], tokenizer, max_length) 

    print('--------------------Predicted--------------------')
    print(f"Predicted caption with 8K MobileNet: {y_pred_Custom}")
    # print(f"Predicted caption with 8K VGG16: {y_pred_8K_VGG16}")
    # print(f"Predicted caption with 8K Resnet50: {y_pred_30K_VGG16}")
    plt.imshow(image)
    plt.show()

generate_caption(image_name)