# remember to install pyqt5 to show images with matplotlib, e.g.  pip3 install PyQt5==5.9.2

import yarp
import time
import sys
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import camera_acquisition

print(sys.argv)

class Module(yarp.RFModule):
    def configure(self, rf):
        # Create ports and connect them to the iCub simulator virtual cameras
        #self.cam_left_port = yarp.Port()
        #self.cam_left_port.open("/python_cam_left_port")
        #self.cam_right_port = yarp.Port()
        #self.cam_right_port.open("/python_cam_right_port")
        #yarp.Network.connect("/icubSim/cam/left", "/python_cam_left_port")
        #yarp.Network.connect("/icubSim/cam/right", "/python_cam_right_port")

        self.camera_grabber_left = camera_acquisition.CameraGrabberExternalWithReader("/icubSim/cam/left", w=640, h=480)
        self.camera_grabber_right = camera_acquisition.CameraGrabberExternalWithReader("/icubSim/cam/right", w=640, h=480)

        # create two subplots
        self.ax_left = plt.subplot(1, 2, 1)
        self.ax_right = plt.subplot(1, 2, 2)

        #self.out_port.open("/testPort/out:o")
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
        np_img_left = self.camera_grabber_left.get_numpy_image()
        np_img_right = self.camera_grabber_right.get_numpy_image()
        # display the image that has been read
        self.ax_left.imshow(np_img_left)
        self.ax_right.imshow(np_img_left)
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