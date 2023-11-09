"""
This module sets up a logger for application-wide use.

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""

import logging

# Configure the logging system
logging.basicConfig(
    format="<%(asctime)s> [%(levelname)s]: %(message)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",  # Optional: Specify date/time format if needed
)

# Create a logger object for the module
logger = logging.getLogger(__name__)
