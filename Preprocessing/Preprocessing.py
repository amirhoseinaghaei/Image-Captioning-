from tensorflow.keras.preprocessing.image import load_img, img_to_array
#from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
#from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import models , Sequential
from tensorflow.keras.models import Model
from tqdm import tqdm 
import os 
import string 
from pickle import load
import numpy as np 



class preprocessing():
	def __init__(self, directory, image_size):
		self.directory = directory
		self.image_size = image_size
	def extract_images(self):
		images = {}
		for name in tqdm(os.listdir(self.directory)):
			self.filename = self.directory + '/' + name
			image = load_img(self.filename, target_size=(self.image_size, self.image_size))
			image = img_to_array(image)
			image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
			image = preprocess_input(image)
			image_id = name.split('.')[0]
			images[image_id] = image
		return images
	def extract_images(self):
		images = {}
		for name in tqdm(os.listdir(self.directory)):
			self.filename = self.directory + '/' + name
			image = load_img(self.filename, target_size=(self.image_size, self.image_size))
			image = img_to_array(image)
			image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
			image = preprocess_input(image)
			image_id = name.split('.')[0]
			images[image_id] = image
		return images
	def extract_features(self):
		model = ResNet50()
		model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
		features = dict()
		for name in tqdm(os.listdir(self.directory)):
			filename = self.directory + '/' + name
			image = load_img(filename, target_size=(224, 224))
			image = img_to_array(image)
			image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
			image = preprocess_input(image)
			feature = model.predict(image, verbose=0)
			image_id = name.split('.')[0]
			features[image_id] = feature
		return features
		
	# load doc into memory
	def load_doc(self, filename):
		# open the file as read only
		file = open(filename, 'r')
		# read all text
		text = file.read()
		# close the file
		file.close()
		return text

	# extract descriptions for images
	def load_descriptions(self, doc):
		mapping = dict()
		# process lines
		for line in doc.split('\n'):
			# split line by white space
			tokens = line.split(",")
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
	

	def clean_descriptions(self, descriptions):
		table = str.maketrans('', '', string.punctuation)
		for key, desc_list in descriptions.items():
			for i in range(len(desc_list)):
	#			desc = desc_list[i]
	#			desc = desc.split()
	#			desc = [word.lower() for word in desc]
	#			desc = [w.translate(table) for w in desc]
	#			desc = [word for word in desc if len(word)>1]
	#			desc = [word for word in desc if word.isalpha()]
	#			desc_list[i] =  ' '.join(desc)
				desc = desc_list[i]
				desc = desc.lower()
				desc = desc.replace('[^A-Za-z]', '')
				desc = desc.replace('\s+', ' ')
				desc = 'startseq ' + " ".join([word for word in desc.split() if len(word)>2]) + ' endseq'
				desc_list[i] = desc
	def to_vocabulary(self, descriptions):
		all_desc = set()
		for key in descriptions.keys():
			[all_desc.update(d.split()) for d in descriptions[key]]
		return all_desc

	def save_descriptions(self, filename,descriptions ):
		lines = list()
		for key, desc_list in descriptions.items():
			for desc in desc_list:
				lines.append(key + ' ' + desc)
		data = '\n'.join(lines)
		file = open(filename, 'w')
		file.write(data)
		file.close()
	# load doc into memory
	def load_doc(self, filename):
		# open the file as read only
		file = open(filename, 'r')
		# read all text
		text = file.read()
		# close the file
		file.close()
		return text

	# load a pre-defined list of photo identifiers
	def load_set(self, filename):
		doc = self.load_doc(filename)
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
	def load_clean_descriptions(self, filename, dataset):
		# load document
		doc = self.load_doc(filename)
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
	def load_photo_features(self,filename, Read, dataset, features):
		# load all features
		all_features = load(open(filename, 'rb')) if Read == False else features
		# filter features
		features = {k: all_features[k] for k in dataset}
		return features

	# load photo images
	def load_photo_images(self,filename, Read, dataset, images):
		# load all images

		all_images = load(open(filename, 'rb')) if Read == False else images
		# all_images = images
		# filter features
		images = {k: all_images[k] for k in dataset}
		return images

	# covert a dictionary of clean descriptions to a list of descriptions
	def to_lines(self, descriptions):
		all_desc = list()
		for key in descriptions.keys():
			[all_desc.append(d) for d in descriptions[key]]
		return all_desc

	# fit a tokenizer given caption descriptions
	def create_tokenizer(self, descriptions):
		lines = self.to_lines(descriptions)
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(lines)
		return tokenizer

	# calculate the length of the description with the most words
	def max_length(self, descriptions):
		lines = self.to_lines(descriptions)
		return max(len(d.split()) for d in lines)


	def create_sequences(self, tokenizer, max_length, descriptions, photos, vocab_size , batch_size):

		X1, X2, y = list(), list(), list()
		n = 0
		for key, desc_list in descriptions.items():
			n += 1
			for desc in desc_list:
				seq = tokenizer.texts_to_sequences([desc])[0]
				for i in range(1, len(seq)):
					in_seq, out_seq = seq[:i], seq[i]
					in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
					out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
					if key != "":
							X1.append((photos[key][0]))
							X2.append(in_seq)
							y.append(out_seq)
			if n == batch_size: 
				X1, X2, y = np.asarray(X1), np.asarray(X2), np.asarray(y)
				yield [X1, X2], y
				X1, X2, y = list(), list(), list()
				n = 0      

		return np.asarray(X1), np.asarray(X2), np.asarray(y)
