
import gzip
import pickle
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import time

x_lims=[0.0,750.0]
x_mean = (x_lims[1] - x_lims[0]) /2.0
y_lims=[0.0,750.0]
y_mean = (y_lims[1]-y_lims[0]) /2.0
z_lims=[-10.0,-90.0]
z_mean = (z_lims[1]-z_lims[0]) /2.0
speed_lim = 3400.0



def normalise_x(x, param):
	if param.get('normalise_with_zero_mean'):
		return  (x - x_mean) / x_lims[1]
	else:
		return x / x_lims[1]

def normalise_y(y, param):
	if param.get('normalise_with_zero_mean'):
		return  (y - y_mean) / y_lims[1]
	else:
		return y / y_lims[1]

def normalise_z(z, param):
	if param.get('normalise_with_zero_mean'):
		return  (z - z_mean) / z_lims[1]
	else:
		return z / z_lims[1]

def normalise(p, param):
	p_n = Position()
	p_n.x = normalise_x(p.x, param)
	p_n.y = normalise_y(p.y, param)
	p_n.z = normalise_z(p.z, param)
	p_n.speed = p.speed
	return p_n

def unnormalise_x(x, param):
	if param.get('normalise_with_zero_mean'):
		return  (x * x_lims[1]) + x_mean
	else:
		return x * x_lims[1]

def unnormalise_y(y, param):
	if param.get('normalise_with_zero_mean'):
		return  (y * y_lims[1]) + y_mean
	else:
		return y * y_lims[1]

def unnormalise_z(z, param):
	if param.get('normalise_with_zero_mean'):
		return  (z * z_lims[1]) + z_mean
	else:
		return z * z_lims[1]

def unnormalise(p, param):
	p_n = Position()
	p_n.x = unnormalise_x(p.x, param)
	p_n.y = unnormalise_y(p.y, param)
	p_n.z = unnormalise_z(p.z, param)
	p_n.speed = p.speed
	return p_n

def clamp(p):
	p_n=Position()
	p_n.x = clamp_x(p.x)
	p_n.y = clamp_y(p.y)
	p_n.z = clamp_z(p.z)
	p_n.speed = p.speed
	return speed

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

def distance (pos_a, pos_b):
	return np.sqrt( np.power(pos_a.x - pos_b.x, 2) + np.power(pos_a.y - pos_b.y, 2) + np.power(pos_a.z - pos_b.z,2) )


def clear_tensorflow_graph():
	print('Clearing TF session')
	if tf.__version__ < "1.8.0":
		tf.reset_default_graph()
	else:
		tf.compat.v1.reset_default_graph()

#### utility functions for reading visuo-motor data from the ROMI dataset
# https://zenodo.org/record/3552827#.Xk5f6hNKjjC
def parse_data( param):
	reshape = param.get('load_data_reshape')
	file_name= param.get('romi_dataset_pkl')
	pixels = param.get('image_size')
	channels = param.get('image_channels')
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

			images.append(np.asarray(image).astype('float32') / 255)

			cmd = memory['command']
			commands.append([normalise_x(float(cmd.x), param), normalise_y(float(cmd.y), param)] )
			#cmd_p = Position()
			#cmd_p.x = float(cmd.x)
			#cmd_p.y = float(cmd.y)
			#cmd_p.z = -90
			#cmd_p.speed = 1400

			#commands.append(normalise(cmd_p))
			pos = memory['position']
			positions.append([normalise_x(float(pos.x), param), normalise_y(float(pos.y), param)] )
			#pos_p = Position()
			#pos_p.x = float(pos.x)
			#pos_p.y = float(pos.y)
			#pos_p.z = -90
			#pos_p.speed = 1400

			#positions.append(normalise(pos_p))

			count += 1

	positions = np.asarray(positions)
	commands = np.asarray(commands)
	images = np.asarray(images)
	if reshape:
		images = images.reshape((len(images), pixels, pixels, channels))
	#print ('images shape ', str(images.shape))
	return images, commands, positions

#### utility functions for reading visuo-motor data from the ROMI dataset
# https://zenodo.org/record/3552827#.Xk5f6hNKjjC
def load_data(param):

	images, commands, positions = parse_data(param)
	# split train and test data
	# set always the same random seed, so that always the same test data are picked up (in case of multiple experiments in the same run)
	np.random.seed(param.get('romi_seed_test_data'))
	test_indexes = np.random.choice(range(len(positions)), param.get('romi_test_size'))
	#reset seed
	np.random.seed(int(time.time()))

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
