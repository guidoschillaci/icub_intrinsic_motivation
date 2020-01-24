# adapted from camera_acquisition.py, written by Lorenzo Vannucci and Murat Kirtay, The BioRobotics Institute, Scuola Superiore Sant'Anna, Pisa, Italy

import yarp
#import rospy
#import scipy.ndimage
import numpy as np
import matplotlib.pylab as plt
#import time

#robot_ip = "localhost"
#robot_port= 9559

#motors = "right_arm"
#props = yarp.Property()

class ICubImageDriver():

	__port_index = 0

	class ImageReader(yarp.PortReader):

		def __init__(self, w, h):
			super(ICubImageDriver.ImageReader, self).__init__()
			# create empty array and image
			self.img_array=np.zeros((h, w), dtype=np.uint8)
			self.yarp_image = yarp.ImageRgb()
			self.yarp_image.resize(w, h)
			self.yarp_image.setExternal(self.img_array, self.img_array.shape[1], self.img_array.shape[0])

		def __del__(self):
			pass

		def read(self, reader):
			self.yarp_image.read(reader)
			return True

		def get_yarp_image(self):
			return self.yarp_image

		def get_numpy_image(self):
			return self.img_array

	def __init__(self, camera_port, width=320, height=240):
		port_name = '/icubRos/image_port'+str(ICubImageDriver.__port_index)
		ICubImageDriver.__port_index =ICubImageDriver.__port_index+1
		# create and connect port
		self.__image_port = yarp.Port()
		self.__image_port.open(port_name)
		yarp.Network.connect(camera_port, port_name)

		self.__reader = ICubImageDriver.ImageReader(width,height)
		self.__image_port.setReader(self.__reader)
		
		print("Driver initialised")

	def __del__(self):
		self.__image_port.close()
	
	def read(self):
		return self.__reader.get_yarp_image()

	def get_numpy_image(self):
		if self.__reader.get_numpy_image() is None:
			return None
		return self.__reader.get_numpy_image()[..., ::-1]

	def show_image(self):
		plt.imshow(self.get_numpy_image())
		plt.show()


if __name__=="__main__":
	yarp.Network.init()
	img_driver = ICubImageDriver("/icubSim/cam/left")
	#img_driver.read()
	img_driver.show_image()
