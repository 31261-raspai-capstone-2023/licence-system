"""
This file defines the logger for use in other files

Authors: Erencan Pelin, Daniel Angeloni, Ben Carroll, Declan Seeto
License: MIT License
Version: 1.0.0
"""
import logging

logging.basicConfig(
    format="<%(asctime)s> [%(levelname)s]: %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)
