FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src

# Copy selected model artifacts
COPY artifacts/470bd3cdf4e541d9946df8164e50b626 ./artifacts/470bd3cdf4e541d9946df8164e50b626

# Set model run ID
ENV MODEL_RUN_ID=470bd3cdf4e541d9946df8164e50b626

# Python output immediately
ENV PYTHONUNBUFFERED=1

# FastAPI port
EXPOSE 8000

# Start FastAPI