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
from keras.callbacks import ModelCheckpoint


# Loading the flicker-dataset images and extracting featueres
from tqdm import tqdm
features = {}
images = {}
img_list = []
directory = os.path.join("Flicker_Dataset", "Images")
# with open(os.path.join(WORKING_DIR , "images.pkl"), "rb") as f:
#     images = pickle.load(f)
for img_name in tqdm(os.listdir(directory)):
  image_path = directory  + "/" + img_name
  image = load_img(image_path , target_size= (128,128))
  image = img_to_array(image) 
  image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
  # feature = model.predict(image)
  # img_list.append(image)
  image_id = img_name.split(".")[0]
  images[image_id] = image
  # features[image_id]  = feature
print(len(images))

from IPython.utils.text import string
def load_captions(path):
  with open(path, 'r') as f: 
    next(f)
    captions = f.read()
  mapping = {}
  line = captions.split("\n")
  for i in line:
    splitted = i.split(",")
    if len(line) < 2:
      continue
    img_id = splitted[0].split(".")[0]
    # print(splitted[1])
    caption = splitted[1:]
    caption = " ".join(caption)
    if img_id not in mapping.keys():
      mapping[img_id] = []
    # print(image_id + ": " + caption)
    mapping[img_id].append(caption)
  return mapping
descriptions = load_captions(os.path.join("Flicker_Dataset", "captions.txt"))
def clean_Descriptions(descriptions):
  table = str.maketrans('','',string.punctuation)
  for key, value in descriptions.items():
    for i in range(len(value)):
      text = descriptions[key][i]
      text = text.lower()
      text = text.translate(str.maketrans('', '', string.punctuation))
      text = text.split()
      text = [ele for ele in text if len(ele) > 1]
      text = [ele for ele in text if (ele).isalpha()]
      text = ' '.join(text)
      descriptions[key][i] = text
  return descriptions
descriptions = clean_Descriptions(descriptions)
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
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
  return (X1), (X2), (y)
# saving features dictionary in pkl file 
from tensorflow.python.keras.layers.merge import Add
from tensorflow.python.ops.gen_array_ops import InplaceAdd
from keras.backend import conv2d

from keras import optimizers
from tensorflow.python.ops.nn_ops import relu
def Build_CNN_FeatureExtractor(vocab_size):
      input1 = Input(shape=(128,128,3))
      conv1 = Conv2D(64, (3, 3), activation= tf.nn.relu, padding="same")(input1)
      conv2 = Conv2D(64, (3, 3), activation= tf.nn.relu, padding="same")(conv1)
      max1 =  MaxPooling2D((2,2), (2,2))(conv2)
    #   conv3 = Conv2D(128, (3, 3), activation= tf.nn.relu, padding="same")(max1)
    #   conv4 = Conv2D(128, (3, 3), activation= tf.nn.relu, padding="same")(conv3)
    #   max2 =  MaxPooling2D((2,2), (2,2))(conv4)
    #   conv5 = Conv2D(256, (3, 3), activation= tf.nn.relu, padding="same")(max2)
    #   conv6 = Conv2D(256, (3, 3), activation= tf.nn.relu, padding="same")(conv5)
    #   conv7 = Conv2D(256, (3, 3), activation= tf.nn.relu, padding="same")(conv6)
    #   max3 =  MaxPooling2D((2,2), (2,2))(conv7)
    #   conv8 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(max3)
    #   conv9 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(conv8)
    #   conv10 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(conv9)
    #   max4 =  MaxPooling2D((2,2), (2,2))(conv10)
    #   conv11 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(max4)
    #   conv12 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(conv11)
    #   conv13 = Conv2D(512, (3, 3), activation= tf.nn.relu, padding="same")(conv12)
    #   max5 =  MaxPooling2D((2,2), (2,2))(conv13)
      flatten = Flatten()(max1)
      dense1 = Dense(500, activation = tf.nn.relu)(flatten)
      dropout1 = Dropout(0.4)(dense1)
    #   dense2 = Dense(1028)(dropout1)
      dense3 = Dense(256)(dropout1)
      input2 = Input(shape = (32,))
      embedding = Embedding(input_dim = vocab_size, output_dim = 256)(input2)
      dropout2 = Dropout(0.4)(embedding)
      lstm = LSTM(256)(dropout2)
      added = add([dense3, lstm])
      dense4 = Dense(256 , activation = tf.nn.relu)(added)
      output = Dense(vocab_size, activation = tf.nn.softmax)(dense4)
      model = Model(inputs = [input1 , input2], outputs = output)
      return model
 
vocab_size = 8765
model = Build_CNN_FeatureExtractor(vocab_size)
optimizer = tf.optimizers.Adam()
model.compile(optimizer = optimizer , loss = "categorical_crossentropy" , metrics = ["Accuracy"] )
model.summary()
import pickle
pickle.dump(images , open(os.path.join("Image_Captioning_Project" , "images.pkl"), "wb"))
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
# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
tokenizer = create_tokenizer(descriptions)
vocab_size = len(tokenizer.word_index) + 1
X1, X2, y = create_sequences(tokenizer, max_length(descriptions), descriptions, images, vocab_size + 1)
Images , Captions, Outputs = np.array(X1), np.array(X2), np.array(y)
X_train1 = Images[0:6900]
X_train2 = Captions[0:6900]
Y_train = Outputs[0:6900]
X_test1 = Images[7500:8091]
X_test2 = Captions[7500:8091]
Y_test = Outputs[7500:8091]
X_val1 = Images[6900:7500]
X_val2 = Captions[6900:7500]
Y_val = Outputs[6900:7500]
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
history = model.fit([X_train1, X_train2], Y_train, epochs=20, verbose=1, callbacks=[checkpoint], validation_data=([X_val1, X_val2], Y_val))