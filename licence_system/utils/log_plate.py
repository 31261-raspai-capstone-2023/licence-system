"""
This module provides utility functions for sending license plate
data and camera location to a front-end web app.

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""

from typing import Optional
import requests
from requests import Response


def make_request(url: str, data: dict, headers: dict) -> Optional[Response]:
    """
    Makes a POST request to the given URL with the provided data and headers.

    Args:
        url (str): The API endpoint to which the request is to be sent.
        data (dict): The payload to be sent in the request.
        headers (dict): The headers to be sent in the request.

    Returns:
        Optional[Response]: The response from the POST request if successful, otherwise None.
    """
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def send_license_plate(license_plate: str, camera_id: int) -> Optional[Response]:
    """
    Sends a license plate number along with the camera ID to a web app API.

    Args:
        license_plate (str): The license plate as a string.
        camera_id (int): The camera ID as an integer.

    Returns:
        Optional[Response]: The response from the POST request if successful, otherwise None.
    """
    api_url = "http://127.0.0.1:5000/add_location_entry"
    data = {"plate": license_plate, "camera_id": camera_id}
    headers = {"Content-Type": "application/json"}

    return make_request(api_url, data, headers)


def add_camera_location(location: str) -> Optional[Response]:
    """
    Sends a camera location to a web app API.

    Args:
        location (str): The location as a string.

    Returns:
        Optional[Response]: The response from the POST request if successful, otherwise None.
    """
    api_url = "http://127.0.0.1:5000/add_camera"
    data = {"location": location}
    headers = {"Content-Type": "application/json"}

    return make_request(api_url, data, headers)
