# remember to install pyqt5 to show images with matplotlib, e.g.  pip3 install PyQt5==5.9.2

import yarp
import cv2
import time
import sys
import numpy as np

import matplotlib.pyplot as plt
import random

print(sys.argv)

class Module(yarp.RFModule):
    def configure(self, rf):

        self.config = yarp.Property()
        self.config.fromConfigFile('/code/icub_intrinsic_motivation/yarp/config.ini')
        self.width = self.config.findGroup('CAMERA').find('width').asInt32()
        self.height = self.config.findGroup('CAMERA').find('height').asInt32()

        if self.config.findGroup('GENERAL').find('show_images').asBool():
            import matplotlib
            matplotlib.use('TKAgg')
            self.ax_left = plt.subplot(1, 2, 1)
            self.ax_right = plt.subplot(1, 2, 2)

        # Create a port and connect it to the iCub simulator virtual camera
        self.input_port_cam = yarp.Port()
        self.input_port_cam.open("/icub/camera_left")
        yarp.Network.connect("/icubSim/cam/left", "/icub/camera_left")

        # prepare image
        self.yarp_img_in = yarp.ImageRgb()
        self.yarp_img_in.resize(self.width, self.height)
        self.img_array = np.ones((self.height, self.width, 3), dtype=np.uint8)
        # yarp image will be available in self.img_array
        self.yarp_img_in.setExternal(self.img_array.data, self.width, self.height)

        # prepare motor driver
        self.left_motors = 'left_arm'
        self.left_motorprops = yarp.Property()
        self.left_motorprops.put("device", "remote_controlboard")
        self.left_motorprops.put("local", "/client/" + self.left_motors)
        self.left_motorprops.put("remote", "/icubSim/" + self.left_motors)

        self.right_motors = 'right_arm'
        self.right_motorprops = yarp.Property()
        self.right_motorprops.put("device", "remote_controlboard")
        self.right_motorprops.put("local", "/client/" + self.right_motors)
        self.right_motorprops.put("remote", "/icubSim/" + self.right_motors)

        # create remote driver
        self.left_armDriver = yarp.PolyDriver(self.left_motorprops)
        print('left motor driver prepared')
        # query motor control interfaces
        self.left_iPos = self.left_armDriver.viewIPositionControl()
        self.left_iVel = self.left_armDriver.viewIVelocityControl()
        self.left_iEnc = self.left_armDriver.viewIEncoders()
        self.left_iCtlLim = self.left_armDriver.viewIControlLimits()

        self.right_armDriver = yarp.PolyDriver(self.right_motorprops)
        print('right motor driver prepared')
        # query motor control interfaces
        self.right_iPos = self.right_armDriver.viewIPositionControl()
        self.right_iVel = self.right_armDriver.viewIVelocityControl()
        self.right_iEnc = self.right_armDriver.viewIEncoders()
        self.right_iCtlLim = self.right_armDriver.viewIControlLimits()

        #  number of joints
        self.num_joints = self.left_iPos.getAxes()
        print('Num joints: ', self.num_joints)

        self.left_limits = []
        self.right_limits = []
        for i in range(self.num_joints):
            left_min =yarp.DVector(1)
            left_max =yarp.DVector(1)
            self.left_iCtlLim.getLimits(i, left_min, left_max)
            print('lim left ', i, ' ', left_min[0], ' ', left_max[0])
            self.left_limits.append([left_min[0], left_max[0]])

            right_min =yarp.DVector(1)
            right_max =yarp.DVector(1)
            self.right_iCtlLim.getLimits(i, right_min, right_max)
            print('lim right ', i, ' ', right_min[0], ' ', right_max[0])
            self.right_limits.append([right_min[0], right_max[0]])

        self.go_to_starting_pos()

        moduleName = rf.check("name", yarp.Value("BabblingModule")).asString()
        self.setName(moduleName)
        print('module name: ',moduleName)
        yarp.delay(1.0)
        print('starting now')

    def close(self):
        print("Going to starting position and closing")
        self.go_to_starting_pos()

    def interruptModule(self):
        return True

    def getPeriod(self):
        return 3 # seconds

    def read_encoders(self):
        print ('Test read encoders')

    def read_skin(self):
        print ('reading skin')


    def read_image(self):
        # read image
        self.input_port_cam.read(self.yarp_img_in)
        # scale down img_array and convert it to cv2 image
        self.image = cv2.resize(self.img_array, (64, 64), interpolation=cv2.INTER_LINEAR)

        if self.config.findGroup('GENERAL').find('show_images').asBool():
            # display the image that has been read
            self.ax_left.imshow(self.image)
            plt.pause(0.01)
            plt.ion()

    def babble_arm(self):

        if not self.left_iPos.checkMotionDone() or not self.right_iPos.checkMotionDone():
            print ('waiting for movement to finish...')
        else:
            print ('new movement')
        target_left_pos =  self.left_startingPos
        target_right_pos =  self.right_startingPos
        for i in range(7,16):
            target_left_pos[i] = random.uniform(self.left_limits[i][0], self.left_limits[i][1])
            target_right_pos[i] = random.uniform(self.right_limits[i][0], self.right_limits[i][1])

        print ('sending command left ', target_left_pos.toString())
        print ('sending command right ', target_right_pos.toString())
        #self.iPos.setControlMode(yarp.Vocab_encode('pos'))
        self.left_iPos.positionMove(target_left_pos.data())
        self.right_iPos.positionMove(target_right_pos.data())
        return target_left_pos, target_right_pos

    def go_to_starting_pos(self):
        # starting position (open hand in front on the left camera
        start_left = yarp.Vector(self.num_joints)
        start_left[0] = -80
        start_left[1] = 16
        start_left[2] = 30
        start_left[3] = 65
        start_left[4] = -5 # was -80
        start_left[5] = 0
        start_left[6] = 0
        start_left[7] = 58.8
        start_left[8] = 20
        start_left[9] = 19.8
        start_left[10] = 19.8
        start_left[11] = 9.9
        start_left[12] = 10.8
        start_left[13] = 9.9
        start_left[14] = 10.8
        start_left[15] = 10.8

        start_right = yarp.Vector(self.num_joints)
        start_right[0] = -80
        start_right[1] = 16
        start_right[2] = 30
        start_right[3] = 65
        start_right[4] = -5 # was -80
        start_right[5] = 0
        start_right[6] = 0
        start_right[7] = 58.8
        start_right[8] = 20
        start_right[9] = 19.8
        start_right[10] = 19.8
        start_right[11] = 9.9
        start_right[12] = 10.8
        start_right[13] = 9.9
        start_right[14] = 10.8
        start_right[15] = 10.8

        self.left_startingPos = yarp.Vector(self.num_joints, start_left.data())
        self.right_startingPos = yarp.Vector(self.num_joints, start_right.data())
        print ('left start : ', self.left_startingPos.toString())
        print('right start : ', self.right_startingPos.toString())
        self.left_iPos.positionMove(self.left_startingPos.data())
        self.right_iPos.positionMove(self.right_startingPos.data())

    # main function, called periodically every getPeriod() seconds
    def updateModule(self):
        self.read_image()
        self.read_skin()
        self.babble_arm()

        return True


yarp.Network.init()


rf = yarp.ResourceFinder()
rf.setVerbose(True);
rf.setDefaultContext("testContext");
rf.setDefaultConfigFile("default.ini");
rf.configure(sys.argv)

mod = Module()
mod.configure(rf)

# check if run calls already configure(rf). upon success, the module execution begins with a call to updateModule()
mod.runModule()

print('after run')