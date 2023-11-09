"""
This module provides functionality to capture images using the PiCamera2.

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""

import time
from picamera2 import Picamera2, Preview


class CameraCapture:
    """
    A CameraCapture class to handle pi camera feed and capture images.

    Attributes:
        cam (Picamera2): The camera instance for capturing images.
    """

    def __init__(self):
        """
        Initializes the CameraCapture instance by setting up the camera
        with a preview configuration.
        """
        self.cam = Picamera2(
            tuning=Picamera2.load_tuning_file(
                "/usr/share/libcamera/ipa/rpi/vc4/imx477_noir.json"
            )
        )
        # Define a preview configuration
        test_config = self.cam.create_preview_configuration({"size": (3280, 2464)})
        self.cam.configure(test_config)

    def capture_pil(self):
        """
        Captures an image using the PiCamera and returns it as a PIL image object.

        Returns:
            Image: The captured image as a PIL object.
        """
        return self.cam.capture_image("main")


def main():
    """
    Main function to execute the camera capture process for testing this code
    """
    camera = CameraCapture()
    camera.cam.start_preview(Preview.QTGL, width=800, height=400)
    camera.cam.start(show_preview=True)

    try:
        for _ in range(5):  # Take 5 photos with a 5-second interval
            time.sleep(5)
            camera.capture_pil()
    finally:
        camera.cam.stop()


if __name__ == "__main__":
    main()
