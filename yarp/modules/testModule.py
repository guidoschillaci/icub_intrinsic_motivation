# remember to install pyqt5 to show images with matplotlib, e.g.  pip3 install PyQt5==5.9.2

import yarp
import cv2
import time
import sys
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

print(sys.argv)

class Module(yarp.RFModule):
    def configure(self, rf, width=320, height=240):

        # create two subplots
        self.ax_left = plt.subplot(1, 2, 1)
        self.ax_right = plt.subplot(1, 2, 2)

        # Create a port and connect it to the iCub simulator virtual camera
        self.input_port_cam = yarp.Port()
        self.input_port_cam.open("/icub/camera_left")
        yarp.Network.connect("/icubSim/cam/left", "/icub/camera_left")

        # prepare image
        self.yarp_img_in = yarp.ImageRgb()
        self.yarp_img_in.resize(width, height)
        self.img_array = np.ones((height, width, 3), dtype=np.uint8)
        # yarp image will be available in self.img_array
        self.yarp_img_in.setExternal(self.img_array.data, width, height)

        # self.out_port.open("/testPort/out:o")
        moduleName = rf.check("name", yarp.Value("MyFirstModule")).asString()
        self.setName(moduleName)
        print('module name: ',moduleName)

    def close(self):
        print("Closing ports")
        #self.cam_left_port.close()
        #self.cam_right_port.close()

    def interruptModule(self):
        #self.cam_left_port.interrupt()
        #self.cam_right_port.interrupt()
        return True

    def getPeriod(self):
        return 1. # seconds

    # main function, called periodically every getPeriod() seconds
    def updateModule(self):
        # Create numpy array to receive the image and the YARP image wrapped around it
        #np_img_left = self.camera_grabber_left.get_numpy_image()
        #np_img_right = self.camera_grabber_right.get_numpy_image()

        # read image
        self.input_port_cam.read(self.yarp_img_in)
        # scale down img_array and convert it to cv2 image
        self.image = cv2.resize(self.img_array, (64, 64), interpolation=cv2.INTER_LINEAR)

        # display the image that has been read
        self.ax_left.imshow(self.image)
        #self.ax_right.imshow(np_img_left)
        plt.pause(0.1)
        plt.ion()
        # Cleanup
        #input_port.close()
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