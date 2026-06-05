FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY modelling.py .
COPY dataset_preprocessing/ ./dataset_preprocessing/

# Expose MLflow port
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["python", "modelling.py"]