import yarp
import rospy
import scipy.ndimage
import numpy as np
import matplotlib.pylab as plt
import time

robot_ip = "localhost"
robot_port= 9559

motors = "right_arm"
props = yarp.Property()

def read_image():
	# Initialise YARP
	yarp.Network.init()
	# Create a port and connect it to the iCub simulator virtual camera
	input_port_cam = yarp.Port()
	input_port_cam.open("/icubRos/image_port")

	input_port_arm = yarp.Port()
	input_port_arm.open("/icubRos/left_arm")

	yarp.Network.connect("/icubSim/cam/left", "/icubRos/image_port")
	yarp.Network.connect("/icubSim/left_arm/analog:o", "/icubRos/left_arm")

	# Create numpy array to receive the image and the YARP image wrapped around it


	yarp_img_in = yarp.ImageRgb()
	yarp_img_in.resize(320,240)
	img_array = bytearray(230400)
	yarp_img_in.setExternal(img_array, 320, 240)
	print(input_port_cam.read(yarp_img_in))
	print(240,' == ',yarp_img_in.height())
	print(320,' == ',yarp_img_in.width())
	img = np.frombuffer(img_array, dtype=np.uint8).reshape(240,320,3)



	plt.imshow(img)
	plt.show()
	# Cleanup
	input_port_cam.close()

if __name__=="__main__":
	read_image()
