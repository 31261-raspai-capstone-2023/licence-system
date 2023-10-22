"""Log car plate to front end"""
from typing import Union

import requests
from requests import Response


def send_license_plate(license_plate: str, camera_id: int) -> Union[Response, bool]:
    """
    Log plate and camera ID to web app API

    Args:
        license_plate (str): The license plate as a string.
        camera_id (int): The camera ID as an integer.

    Returns:
        requests.Response: The response from the POST request.
    """
    # Define the API URL
    api_url = "http://127.0.0.1:5000/add_location_entry"

    # Create a dictionary with the data to be sent in the request
    data = {"plate": license_plate, "camera_id": camera_id}

    # Set the headers for the request
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
