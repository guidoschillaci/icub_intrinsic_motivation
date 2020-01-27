#!/usr/bin/env python
#from _ast import alias
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from icub_drivers.msg import Commands, JointPositions
import data_keys
import time
import yarp
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()

import os
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
os.environ["ROS_HOSTNAME"] = "localhost"

#robot_ip = "192.168.26.135"
robot_ip = "localhost"
#robot_port = 9559

class IntrinsicMotivation():
	def __init__(self):
		# flag to check for generating new motor commands
		self.is_moving = False

		# create ros node
		rospy.init_node('icub_intrinsic_motivation', anonymous=True)

		# create the publishers for sending motor commands
		self.cmd_pub = rospy.Publisher('/icubRos/commands/move_to_joint_pos', JointPositions, queue_size=10)
		
		# subscribe to the topics
		joint_speed_sub = rospy.Subscriber('/icubRos/sensors/joint_speeds', JointPositions, self.joint_speed_callback, queue_size=10)

		rospy.spin()
		#self.motor_babbling()

	def joint_speed_callback(self, speed_msg):
		# do it for all the joint groups
		speeds = speed_msg.head
		#print (str(speeds))
		if sum(speeds[0:2])< data_keys.SPEED_THRESHOLD / 3:
			self.is_moving=False
			print ("not moving")
		else:
			self.is_moving = True
			print ("moving")

	def motor_babbling(self):
		while True:

			pass

if __name__=="__main__":
	rospy.loginfo("intrinsic_motivation")
	try:
		intrMot = IntrinsicMotivation() 
		
	except rospy.ROSInterruptException:
		pass
