
import gzip
import pickle
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import time

x_lims=[0.0,750.0]
y_lims=[0.0,750.0]
z_lims=[-10.0,-90.0]
speed_lim = 3400.0

def clamp_x(x):
	if x <= x_lims[0]:
		return x_lims[0]
	if x > x_lims[1]:
		return x_lims[1]
	return x

def clamp_y(y):
	if y <= y_lims[0]:
		return y_lims[0]
	if y > y_lims[1]:
		return y_lims[1]
	return y

def clamp_z(z):
	if z <= z_lims[0]:
		return z_lims[0]
	if z > z_lims[1]:
		return z_lims[1]
	return z


class Position:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.speed = 0


def clear_tensorflow_graph():
	print('Clearing TF session')
	if tf.__version__ < "1.8.0":
		tf.reset_default_graph()
	else:
		tf.compat.v1.reset_default_graph()

#### utility functions for reading visuo-motor data from the ROMI dataset
# https://zenodo.org/record/3552827#.Xk5f6hNKjjC
def parse_data( file_name, pixels, reshape, channels=1):
	images = []
	positions = []
	commands = []
	with gzip.open(file_name, 'rb') as memory_file:
		memories = pickle.load(memory_file)
		print ('converting data...')
		count = 0
		for memory in memories:
			image = memory['image']
			if (channels == 1) and (image.ndim == 3):
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			image = cv2.resize(image, (pixels, pixels))

			images.append(np.asarray(image))

			cmd = memory['command']
			commands.append([float(cmd.x) / x_lims[1], float(cmd.y) / y_lims[1]])
			pos = memory['position']
			positions.append([float(pos.x) / x_lims[1], float(pos.y) / y_lims[1]])

			count += 1

	positions = np.asarray(positions)
	commands = np.asarray(commands)
	images = np.asarray(images)
	if reshape:
		images = images.reshape((len(images), pixels, pixels, channels))
	#print ('images shape ', str(images.shape))
	return images.astype('float32') / 255, commands, positions

#### utility functions for reading visuo-motor data from the ROMI dataset
# https://zenodo.org/record/3552827#.Xk5f6hNKjjC
def load_data( dataset, image_size, param):

	images, commands, positions = parse_data(dataset, reshape=True, pixels = image_size)
	# split train and test data
	# set always the same random seed, so that always the same test data are picked up (in case of multiple experiments in the same run)
	np.random.seed(param.get('romi_seed_test_data'))
	test_indexes = np.random.choice(range(len(positions)), param.get('test_size'))
	#reset seet
	np.random.seed(time.time())

	# print ('test idx' + str(test_indexes))
	train_indexes = np.ones(len(positions), np.bool)
	train_indexes[test_indexes] = 0

	# split images
	test_images = images[test_indexes]
	train_images = images[train_indexes]
	test_cmds = commands[test_indexes]
	train_cmds = commands[train_indexes]
	test_pos = positions[test_indexes]
	train_pos = positions[train_indexes]
	print ("number of train images: ", len(train_images))
	print ("number of test images: ", len(test_images))

	return train_images, test_images, train_cmds, test_cmds, train_pos, test_pos

# get NN layer index by name
def getLayerIndexByName(model, layername):
	for idx, layer in enumerate(model.layers):
		if layer.name == layername:
			return idx
