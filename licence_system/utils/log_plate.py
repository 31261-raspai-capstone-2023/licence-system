"""
This file defines all the utility functions for logging a plate to the front end web app

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
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

    # try:
    response = requests.post(api_url, json=data, headers=headers, timeout=10)
    response.raise_for_status()
    return response
    # except requests.exceptions.RequestException as e:
    #     print(f"An error occurred: {e}")
    #     return None


def add_camera_location(location: str) -> Union[Response, bool]:
    """
    Log the location to the web app API

    Args:
        location (str): The location as a string.

    Returns:
        requests.Response: The response from the POST request if successful, otherwise False.
    """
    # Define the API URL
    api_url = "http://127.0.0.1:5000/add_camera"

    # Create a dictionary with the data to be sent in the request
    data = {"location": location}

    # Set the headers for the request
    headers = {"Content-Type": "application/json"}

    try:
        # Send a POST request
        response = requests.post(api_url, json=data, headers=headers, timeout=10)
        response.raise_for_status()  # This will raise an error for HTTP error codes
        return response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return False