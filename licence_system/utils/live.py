"""
This file takes a photo

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""

from picamera2 import Picamera2, Preview
# from libcamera import Transform
from PIL import Image
import datetime
import time

class CameraCapture:
    def __init__(self):
        # tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/vc4/imx477_noir.json") # define the tuning file to account for the lack of an infrared filter
        self.cam = Picamera2(tuning=Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/vc4/imx477_noir.json")) # load the tuning filter
        test_config = self.cam.create_preview_configuration({ # define a preview configuration 
            "size": (3280,2464),
        })
        # capture_config = cam.create_still_configuration()
        self.cam.configure(test_config) # load in the configuration defined above

    def capture_pil(self):
        return self.cam.capture_image("main")
    
if __name__ == "__main__":
    testclass = CameraCapture()
    testclass.cam.start_preview(Preview.QTGL, width=800, height=400) # configure the preview window
    testclass.cam.start(show_preview=True) # open the preview window

    for x in range(0,5): # continually take photos every 5 seconds
        time.sleep(5)
        # cam.switch_mode_and_capture_file(capture_config, f"live_images/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jpg")
        image = testclass.cam.capture_image("main")

    testclass.cam.stop()
    # wrap in a class
    # in init - something actual recording
    # the code that captures the image - separate function