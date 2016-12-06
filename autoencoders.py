import utils
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import json
import math
import numpy as np


### A simple autoencoder model ------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------ #

# dimension of input
input_dim = 175

# this is the size of our encoded representations
encoding_dim = 40  

# this is our input placeholder
input_objects = Input(shape=(input_dim,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_objects)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_objects, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_objects, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mse')
encoder.compile(optimizer='adadelta', loss='mse')
decoder.compile(optimizer='adadelta', loss='mse')

# ---------------------------------------------------------------------------------------- #
# train the model ------------------------------------------------------------------------ #

print ('Beginning training ... ')
load_easy = False
load_small = False
train_data = utils.get_object_multihots('train', load_easy, load_small)
val_data = utils.get_object_multihots('val', load_easy, load_small)
all_data = np.concatenate((train_data, val_data), axis=0)
all_data = np.array(filter(lambda x: sum(x)!=0, all_data))

batch_size = 128

autoencoder.fit(all_data, all_data,
		            nb_epoch=20,
		            batch_size=batch_size,
		            shuffle=True)

print ('Done training ... ')

# ---------------------------------------------------------------------------------------- #


print ('Beginning to save compressions...')

compressed_object_dictionary = {}

print all_data[0].shape

for obj in all_data: 
	new_obj = obj.reshape(obj.shape[0], 1)
	encoded_object = encoder.predict(np.transpose(new_obj))
	compressed_object_dictionary[str(obj)] = encoded_object.tolist()

print ('Done!')

with open('compressed_object_dictionary.txt', 'w') as outfile: 
	json.dump(compressed_object_dictionary, outfile, sort_keys=True, indent=4)

print ('Saved to compressed_object_dictionary.txt')
