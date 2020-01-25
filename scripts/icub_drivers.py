import yarp
import rospy
import scipy.ndimage
import numpy as np
import matplotlib.pylab as plt
import time

#motors = "right_arm"
#props = yarp.Property()

ros_ip = "localhost"
ros_port= 9559


class ImageReader():
	
	def __init__(self, width=640, height=480):

		# Create a port and connect it to the iCub simulator virtual camera
		self.input_port_cam = yarp.Port()
		self.input_port_cam.open("/icubRos/image_port")

		yarp.Network.connect("/icubSim/cam/left", "/icubRos/image_port")

		# prepare image
		self.yarp_img_in = yarp.ImageRgb()
		self.yarp_img_in.resize(width,height)
		self.img_array = np.zeros((height, width, 3), dtype=np.uint8)
		self.yarp_img_in.setExternal(self.img_array, width, height)

		#self.yarp_img_in.setExternal(self.img_array.__array_interface__['data'][0], width, height)

		
	def read_image(self):

		# Create numpy array to receive the image and the YARP image wrapped around it
		print(self.input_port_cam.read(self.yarp_img_in))
		print(240,' == ',self.yarp_img_in.height())
		print(320,' == ',self.yarp_img_in.width())
		#img = np.frombuffer(self.img_array, dtype=np.uint8).reshape(self.yarp_img_in.height(),self.yarp_img_in.width(),3)

		plt.imshow(self.img_array)
		plt.show()

	def __del__(self):
		# Cleanup
		self.input_port_cam.close()
		

class JointReader():
	def __init__(self):

		self.motor_string="right_arm"
		self.props = yarp.Property()
		self.props.put("device", "remote_controlboard")
		self.props.put("local", "/client/"+self.motor_string)
		self.props.put("remote", "/icubSim/"+self.motor_string )

		self.arm_driver = yarp.PolyDriver(self.props)
		# creating interfaces
		self.iPos = self.arm_driver.viewIPositionControl()
		self.iVel = self.arm_driver.viewIVelocityControl()
		self.iEnc = self.arm_driver.viewIEncoders()

		yarp.delay(1.0)
		# number ofjoints
		self.n_joints = self.iPos.getAxes()
		print ("number of joints: ", self.n_joints)


		#self.input_port_arm = yarp.Port()
		#self.input_port_arm.open("/icubRos/left_arm")
		#yarp.Network.connect("/icubSim/left_arm/analog:o", "/icubRos/left_arm")

	def read_position(self):	
		self.current_position = yarp.Vector(self.n_joints)
		self.iEnc.getEncoders(self.current_position.data())
		#data = list(map(self.current_position.data().__get_item__,range(self.current_position.data().size())))
		print("current pos ", self.current_position.toString(-1,1))

	def __del__(self):
		# Cleanup
		#self.input_port_arm.close()
		pass


if __name__=="__main__":

	# Initialise YARP
	yarp.Network.init()		

	#read_ijoints()
	jointReader = JointReader()
	i=0
	while i<10:
		jointReader.read_position()
		i=i+1

	imageReader = ImageReader()
	i=0
	while i<4:
		imageReader.read_image()
		i=i+1
