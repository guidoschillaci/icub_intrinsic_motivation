#! /usr/bin/env python

#from enum import Enum
import numpy as np
from icub_drivers.msg import JointPositions

JointNames = ["head",
#"torso",
"left_arm",
#"left_hand",
#"left_hand_thumb",
#"left_hand_index",
#"left_hand_middle",
#"left_hand_finger",
#"left_hand_pinky",
#"left_foot",
#"right_arm",
#"right_hand",
#"right_hand_thumb",
#"right_hand_index",
#"right_hand_middle",
#"right_hand_finger",
#"right_hand_pinky",
#"right_foot"
]

def set_joint_pos_msg_value(_msg, attr, _value, size):
	print ('before ',_value)
	value =np.fromstring( _value, dtype=float, sep=' ')
	print (str(value))
	msg=_msg
	if attr == "head": msg.head = value
	if attr == "torso": msg.torso = value
	if attr == "left_arm": msg.left_arm = value
	if attr == "left_hand": msg.left_hand = value
	if attr == "left_hand_thumb": msg.left_hand_thumb = value
	if attr == "left_hand_index": msg.left_hand_index = value
	if attr == "left_hand_middle": msg.left_hand_middle = value
	if attr == "left_hand_finger": msg.left_hand_finger = value
	if attr == "left_hand_pinky": msg.left_hand_pinky = value
	if attr == "left_foot": msg.left_foot = value

	if attr == "right_arm": msg.right_arm = value
	if attr == "right_hand": msg.right_hand = value
	if attr == "right_hand_thumb": msg.right_hand_thumb = value
	if attr == "right_hand_index": msg.right_hand_index = value
	if attr == "right_hand_middle": msg.right_hand_middle = value
	if attr == "right_hand_finger": msg.right_hand_finger = value
	if attr == "right_hand_pinky": msg.right_hand_pinky = value
	if attr == "right_foot": msg.right_foot = value

	return msg

##############################

'''
class JointEnum (Enum):
    RShoulderRoll=0
    RShoulderPitch=1
    RElbowYaw=2
    RElbowRoll=3
    RWristYaw=4
    LShoulderRoll=5
    LShoulderPitch=6
    LElbowYaw=7
    LElbowRoll=8
    LWristYaw=9
    Time=10

JointLimits = np.asarray([[-1.5620,-0.0087],
		[-2.0857, 2.0857],
		[-2.0857, 2.0857],
		[0.0087, 1.5620],
		[-1.8239, 1.8239],
		[0.0087, 1.5620],
		[-2.0857, 2.0857],
		[-2.0857, 2.0857],
		[-1.5620, -0.0087],
		[-1.8239, 1.8239]])

class SensorEnum (Enum):
    SensorValues=0
    ActuatorValues=1
    ElectricCurrentValues=2
    TemperatureValues=3
    TemperatureStatus=4
    HardnessValues=5

SensorNames = ["SensorValues",
             "ActuatorValues",
             "ElectricCurrentValues",
             "TemperatureValues",
             "TemperatureStatus",
             "HardnessValues"]

'''

