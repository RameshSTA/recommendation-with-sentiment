# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . /app

# Define environment variable
ENV PORT=8080

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define the command to run your application when the container starts
CMD [ "python", "main.py" ] # Replace "main.py" with the entry point of your application
