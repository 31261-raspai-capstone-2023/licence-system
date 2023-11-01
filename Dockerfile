# Use an official Python runtime as a base image
FROM python:3.11-slim-buster

# Set the working directory in the container to /app
WORKDIR /licence_system

# Copy the current directory contents into the container at /app
COPY . /licence_system

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "-um", "licence_system.main2"]

