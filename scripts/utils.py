
import gzip
import pickle
import cv2
import numpy as np
import tensorflow as tf

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
			if channels == 1:
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
def load_data( dataset, image_size, step):

	images, commands, positions = parse_data(dataset, reshape=True, pixels = image_size)
	# split train and test data
	indices = range(0, len(positions), step) ## TODO: choose random samples instead!
	# split images
	test_images = images[indices]
	train_images = images
	test_cmds = commands[indices]
	train_cmds = commands
	test_pos = positions[indices]
	train_pos = positions
	print ("number of train images: ", len(train_images))
	print ("number of test images: ", len(test_images))

	return train_images, test_images, train_cmds, test_cmds, train_pos, test_pos

# get NN layer index by name
def getLayerIndexByName(model, layername):
	for idx, layer in enumerate(model.layers):
		if layer.name == layername:
			return idx
