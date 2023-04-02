import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Conv2D , MaxPooling2D   , Flatten
from tensorflow.keras.models import Model

class Custum_CNN_LSTM:
    def __init__(self , vocab_size , image_size, max_length):
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.max_length = max_length
    def Build_CNN_FeatureExtractor(self):
        input1 = Input(shape=self.image_size)
        conv1 = Conv2D(64, (3, 3), activation= tf.nn.relu, padding="same")(input1)
        max1 =  MaxPooling2D((2,2), (2,2))(conv1)
        conv3 = Conv2D(128, (3, 3), activation= tf.nn.relu, padding="same")(max1)
        max2 =  MaxPooling2D((2,2), (2,2))(conv3)
        flatten = Flatten()(max2)
        dense1 = Dense(1028, activation = tf.nn.relu)(flatten)
        dropout1 = Dropout(0.4)(dense1)
        dense3 = Dense(256)(dropout1)
        input2 = Input(shape = (self.max_length,))
        embedding = Embedding(input_dim = self.vocab_size, output_dim = 256)(input2)
        dropout2 = Dropout(0.4)(embedding)
        lstm = LSTM(256)(dropout2)
        added = add([dense3, lstm])
        dense4 = Dense(256 , activation = tf.nn.relu)(added)
        output = Dense(self.vocab_size, activation = tf.nn.softmax)(dense4)
        model = Model(inputs = [input1 , input2], outputs = output)
        model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics = ["accuracy"])
        return model