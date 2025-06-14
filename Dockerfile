# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy only requirements first for better build caching
COPY requirements.txt ./

# Install pip and dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Now copy the rest of your code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 80

# Start FastAPI with Uvicorn
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "80"]
