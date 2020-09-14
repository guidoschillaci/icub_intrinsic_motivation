print('before declaring class')

import yarp
import time
import sys
import numpy as np
import matplotlib

print(sys.argv)

class Module(yarp.RFModule):
    def configure(self, rf):
        # Create ports and connect them to the iCub simulator virtual cameras
        self.cam_left_port = yarp.Port()
        self.cam_left_port.open("/python_cam_left_port")
        self.cam_right_port = yarp.Port()
        self.cam_right_port.open("/python_cam_right_port")
        yarp.Network.connect("/icubSim/cam/left", "/python_cam_left_port")
        yarp.Network.connect("/icubSim/cam/right", "/python_cam_right_port")

        self.out_port.open("/testPort/out:o")
        moduleName = rf.check("iCubTest", yarp.Value("iCubTest")).asString()
        self.setName(moduleName)
        print(moduleName)

    def close(self):
        print("Closing ports")
        self.cam_left_port.close()
        self.cam_right_port.close()

    def interruptModule(self):
        self.cam_left_port.interrupt()
        self.cam_right_port.interrupt()
        return True

    def getPeriod(self):
        return 1. # seconds
    # main function, called periodically every getPeriod() seconds
    def updateModule(self):

        # Create numpy array to receive the image and the YARP image wrapped around it
        img_array = np.zeros((240, 320, 3), dtype=np.uint8)
        yarp_image = yarp.ImageRgb()
        yarp_image.resize(320, 240)
        yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])

        # Alternatively, if using Python 2.x, try:
        # yarp_image.setExternal(img_array.__array_interface__['data'][0], img_array.shape[1], img_array.shape[0])

        # Read the data from the port into the image
        self.cam_left_port.read(yarp_image)

        # display the image that has been read
        matplotlib.pylab.imshow(img_array)

        # Cleanup
        #input_port.close()
        return True


yarp.Network.init()

mod = Module()

rf = yarp.ResourceFinder()
rf.setVerbose(True);
rf.setDefaultContext("myContext");
rf.setDefaultConfigFile("default.ini");
rf.configure(sys.argv)

mod.configure(rf)

# check if run calls already configure(rf). upon success, the module execution begins with a call to updateModule()
mod.runModule(rf)

print('after run')