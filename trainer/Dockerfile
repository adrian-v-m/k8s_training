# Use an official Pytorch runtime as a parent image
FROM nvcr.io/nvidia/pytorch:25.02-py3

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python package and its source code into the container
COPY . /app

# Run the model_train_lora.py script when the container launches
ENTRYPOINT ["torchrun", "model_train_lora.py"] 