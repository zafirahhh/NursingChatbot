# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all local files to the container
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download required NLTK data
RUN python -m nltk.downloader punkt

# Expose the port required by Hugging Face (7860)
EXPOSE 7860

# Start the FastAPI app using uvicorn on the correct port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
