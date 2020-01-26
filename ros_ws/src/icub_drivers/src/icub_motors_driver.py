#!/usr/bin/env python
from _ast import alias
import numpy as np
import rospy
from pepper_drivers.msg import MotorCommand, SensorsValues
from std_msgs.msg import Empty
import qi # naoqi
import sys
import data_keys
import signal

import os
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
os.environ["ROS_HOSTNAME"] = "localhost"


class PepperMotorDriver:

	def __init__(self):
		self.robot_ip = "192.168.26.135"
		self.robot_port = 9559

		signal.signal(signal.SIGINT, self.Exit_call)
		self.exitCallFlag = False

		# start the naoqi session
		self.session = qi.Session()
		try:
			# try to connect to the robot
			rospy.loginfo("Connecting to naoqi")
			self.session.connect("tcp://" + self.robot_ip + ":" + str(self.robot_port))
		except RuntimeError:
			#print "Can't connect to Naoqi at ip \"", robot_ip, "\" on port ", str(robot_port)
			rospy.logerr("Can't connect to Naoqi at ip %s on port %s", self.robot_ip, self.robot_port)
			sys.exit(1) # check how to terminate the execution properly in ros!!!!
		rospy.loginfo("connected to naoqi")

		# create ros node
		rospy.init_node('pepper_motors_driver', anonymous=True)

		# create the subscribers
		move_to_joint_sub = rospy.Subscriber('/pepper/commands/move_to_joint_pos', MotorCommand, self.move_to_joint_pos_callback, queue_size=10)
		reach_target_electric_current_sub = rospy.Subscriber('/pepper/commands/reach_target_electric_current', SensorsValues, self.reach_target_electric_current_callback, queue_size=10)
		rest_sub = rospy.Subscriber('/pepper/commands/rest', Empty, self.rest_callback, queue_size=10)
		stand_zero_sub = rospy.Subscriber('/pepper/commands/stand_zero', Empty, self.stand_zero_callback, queue_size=10)
		stand_init_sub = rospy.Subscriber('/pepper/commands/stand_init', Empty, self.stand_init_callback, queue_size=10)

		# initialise ROS motor command message to None
		self.target_joint_position = None
		self.target_electric_current = None

		# if the naoqi connection is working
		if self.session.isConnected():
			print 'is connected'
			# create the proxy to the motion module
			self.motion_service = self.session.service("ALMotion")
			self.posture_service = self.session.service("ALRobotPosture")
			self.alife_service = self.session.service("ALAutonomousLife")
			self.alife_service.setAutonomousAbilityEnabled("All", False)

			# Wake up robot
			self.motion_service.wakeUp()
			# Send robot to Stand Zero
			self.posture_service.goToPosture("StandZero", 0.5)

		else:
			print 'is not connected'

	def move_to_joint_pos_callback(self, msg):

		self.target_joint_position = msg

		angles = [msg.RShoulderRoll,
				  msg.RShoulderPitch,
				  msg.RElbowYaw,
				  msg.RElbowRoll,
				  msg.RWristYaw,
				  msg.LShoulderRoll,
				  msg.LShoulderPitch,
				  msg.LElbowYaw,
				  msg.LElbowRoll,
				  msg.LWristYaw]

		
		if (msg.fractionMaxSpeed>=0 or msg.fractionMaxSpeed<=1):
			#print 'driver joints ', data_keys.JointNames ,' cmd ' , angles, ' fractionMaxSpeed ', msg.fractionMaxSpeed
			self.motion_service.setAngles(data_keys.JointNames, angles, msg.fractionMaxSpeed)


	def reach_target_electric_current_callback(self, msg):
		self.target_electric_current = msg

	def rest_callback(self, msg):
		self.motion_service.rest()

	def stand_zero_callback(self, msg):
		self.posture_service.goToPosture("StandZero", 0.5)

	def stand_init_callback(self, msg):
		self.posture_service.goToPosture("StandInit", 0.5)


	def Exit_call(self, signal, frame): # this is called when typing CTRL+c from the terminal
		if not self.exitCallFlag:
			self.exitCallFlag = True
			# Go to rest position
			self.motion_service.rest()
			print "script terminated!"
			sys.exit(1)

	def run_driver(self):
		self.cmd_pub = rospy.Publisher('/pepper/targets/target_joint_position', MotorCommand, queue_size=10)
		self.target_elec_current_pub = rospy.Publisher('/pepper/targets/target_electric_current', SensorsValues, queue_size=10)

		if self.session.isConnected():

			# do x iterations per second (do not forget to put rate.sleep at the end of the while loop)
			rate = rospy.Rate(10) # Hz
			while not rospy.is_shutdown():
                                # publish a copy of the most recently requested target joint position
				if self.target_joint_position is not None:
					self.target_joint_position.header.stamp = rospy.Time.now()
					self.cmd_pub.publish(self.target_joint_position)
					
                                # publish a copy of the most recently requested target electric current
				if self.target_electric_current is not None:
					self.target_electric_current.header.stamp = rospy.Time.now()
					self.target_elec_current_pub.publish(self.target_electric_current)
					
				# sleep (function of rospy.Rate(x))
				rate.sleep()



if __name__=="__main__":
	rospy.loginfo("main")
	try:
		pmd = PepperMotorDriver()
		pmd.run_driver()
	except rospy.ROSInterruptException:
		pass

