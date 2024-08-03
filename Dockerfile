# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=127.0.0.1

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]
