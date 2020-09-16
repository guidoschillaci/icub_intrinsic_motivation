# Authors:
# Lorenzo Vannucci, The BioRobotics Institute, Scuola Superiore Sant'Anna, Pisa, Italy
# Murat Kirtay, The BioRobotics Institute, Scuola Superiore Sant'Anna, Pisa, Italy

import yarp
import numpy as np


class CameraGrabber(object):

    def grab_yarp_image(self):
        """
        Grabs the next image from a YARP port and returns as it is (yarp.ImageRgb).
        In general, this method should be called before get_numpy_img.
        """
        raise NotImplementedError

    def get_numpy_image(self):
        """
        Returns the last image read as a numpy array (GBR)
        """
        raise NotImplementedError


class CameraGrabberBuffered(CameraGrabber):
    """
    This grabber uses BufferedPort, thus is completely decoupled from the sender. This is in principle the best option,
    but converting the image into numpy array is currently very slow (> 1s). Thus, it is not advisable to use this
    unless one needs only the yarp image.
    """

    __portindex = 0

    def __init__(self, camera_port, w=640, h=480):

        # create unique name for port
        portname = '/icubpython/cameragrabberbuf' + str(CameraGrabberBuffered.__portindex)
        CameraGrabberBuffered.__portindex = CameraGrabberBuffered.__portindex + 1

        # create and connect port
        self.__imageport = yarp.BufferedPortImageRgb()
        self.__imageport.open(portname)
        yarp.Network.connect(camera_port, portname)

        self.__yarp_image = None
        self.__camw = w
        self.__camh = h

    def __del__(self):
        self.__imageport.close()

    def grab_yarp_image(self):
        self.__yarp_image = self.__imageport.read(False)
        return self.__yarp_image

    def get_numpy_image(self):
        """
        This converts the image into a new numpy array (very slow)
        """
        if self.__yarp_image is None:
            return None

        img_array = np.zeros((self.__camh, self.__camw, 3), dtype=np.uint8)
        for x in xrange(self.__camh):
            for y in xrange(self.__camw):
                p = self.__yarp_image.pixel(y, x)
                img_array[x][y][0] = p.b
                img_array[x][y][1] = p.g
                img_array[x][y][2] = p.r

        return img_array


class CameraGrabberExternal(CameraGrabber):
    """
    This grabber uses port with preallocated external memory for the image. Thus, there are zero copies, but, it uses
    Port, so it will be coupled with the sender. Thus, the sender will probably be slowed down. This should only be used
    with a high reading frequency.
    """

    __portindex = 0

    def __init__(self, camera_port, w=640, h=480):

        # create unique name for port
        portname = '/icubpython/cameragrabberext' + str(CameraGrabberExternal.__portindex)
        CameraGrabberExternal.__portindex = CameraGrabberExternal.__portindex + 1

        # create and connect port
        self.__imageport = yarp.Port()
        self.__imageport.open(portname)
        yarp.Network.connect(camera_port, portname)

        # pre-allocation of memory to save time
        self.__img_array = np.zeros((h, w, 3), dtype=np.uint8)
        self.__yarp_image = yarp.ImageRgb()
        self.__yarp_image.resize(w, h)
        self.__yarp_image.setExternal(self.__img_array, self.__img_array.shape[1], self.__img_array.shape[0])

    def __del__(self):
        self.__imageport.close()

    def grab_yarp_image(self):
        self.__imageport.read(self.__yarp_image)
        return self.__yarp_image

    def get_numpy_image(self):
        """
        This returns the underlying numpy array of the image (just reordering channels)
        """
        if self.__yarp_image is None:
            return None
        return self.__img_array[..., ::-1]


class CameraGrabberExternalWithReader(CameraGrabber):
    """
    This grabber uses a PortReader to decouple sender and receiver, in addition to the external allocated memory.
    Currently, this is the best option. However, things such as threading issues have not been investigated and may
    cause some problems.
    """

    __portindex = 0

    class ImageReader(yarp.PortReader):

        def __init__(self, w, h):

            super(CameraGrabberExternalWithReader.ImageReader, self).__init__()
            # pre-allocation of memory to save time
            self.__img_array = np.zeros((h, w, 3), dtype=np.uint8)
            self.__yarp_image = yarp.ImageRgb()
            self.__yarp_image.resize(w, h)

            self.__yarp_image.setExternal(self.__img_array.data, self.__img_array.shape[1], self.__img_array.shape[0])

        def __del__(self):
            pass

        def read(self, reader):
            # reader.read(self.__yarp_image)
            self.__yarp_image.read(reader)
            return True

        def get_yarp_img(self):
            return self.__yarp_image

        def get_numpy_img(self):
            return self.__img_array

    def __init__(self, camera_port, w=640, h=480):

        super(CameraGrabberExternalWithReader, self).__init__()

        # create unique name for port
        portname = '/icubpython/cameragrabberextread' + str(CameraGrabberExternalWithReader.__portindex)
        CameraGrabberExternalWithReader.__portindex = CameraGrabberExternalWithReader.__portindex + 1

        # create and connect port
        self.__imageport = yarp.Port()
        self.__imageport.open(portname)
        yarp.Network.connect(camera_port, portname)
        # reader
        self.__reader = CameraGrabberExternalWithReader.ImageReader(w, h)
        self.__imageport.setReader(self.__reader)


    def __del__(self):
        self.__imageport.close()

    def grab_yarp_image(self):
        """
        This actually just returns the last image read by the reader, as reading is asynchronous.
        There is no need to call this method before get_numpy_img, for this class.
        """
        return self.__reader.get_yarp_img()

    def get_numpy_image(self):
        """
        This returns the underlying numpy array of the last image read by the reader (just reordering channels).
        Can be called without calling grab_yarp_image
        """
        if self.__reader.get_numpy_img() is None:
            return None
        return self.__reader.get_numpy_img()[..., ::-1]
