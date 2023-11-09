# Use an official Python runtime as a base image
FROM python:3.11-slim-buster

# Set the working directory in the container to /licence_system
WORKDIR /licence_system

# Copy the current directory contents into the container at /licence_system
COPY . /licence_system

# Install Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    python3-picamera2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the Tesseract path
ENV TESSERACT_CMD /usr/bin/tesseract

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the python file when the container launches
CMD ["python", "-um", "licence_system.main"]