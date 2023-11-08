"""
This file defines the runner for inferencing the model

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import faulthandler
import time
import numpy as np
import os
from licence_system.utils.model_class import LPR_Inference
from licence_system.utils.live import CameraCapture
from licence_system.utils.ocr import ocr_image
from licence_system.utils.log_plate import send_license_plate

faulthandler.enable()

# # Retrieve the LPR_CAMERA_ID environment variable
# CAMERA_ID = os.getenv("LPR_CAMERA_ID")

# # If the environment variable is not set, raise an exception
# if CAMERA_ID is None:
#     raise EnvironmentError("The TESSERACT_CMD environment variable is not set.")

inference_class = LPR_Inference(
    model_path="licence_system/models/checkpoints/LPLocalNet_B250_E500_LR0.0010_Acc74.22.pth",
    display_output=True,
    bbox_buffer=60,
)

if __name__ == "__main__":
    camera_class = CameraCapture()

    try:
        camera_class.cam.start()  # open the preview window

        # Run indefinitely until a keyboard interrupt (Ctrl+C) or other exception occurs
        while True:
            image = inference_class.get_bounding_box_from_img(
                camera_class.capture_pil()
            )
            image = np.asarray(image)

            # Run OCR
            licence_plate = ocr_image(image)
            print(licence_plate)

            # # Log plate to front end
            # send_license_plate(licence_plate, CAMERA_ID)
            
            time.sleep(0.1)  # Add a short delay to not overwork the CPU

    except KeyboardInterrupt:
        print("Interrupted by user, stopping...")
    finally:
        camera_class.cam.stop()  # Ensure the camera is stopped
