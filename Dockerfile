FROM python:3.11-slim-buster

# Update system
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install ageml

ENTRYPOINT [ "/bin/bash" ]