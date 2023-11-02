import logging

logging.basicConfig(
    format="<%(asctime)s> [%(levelname)s]: %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)
