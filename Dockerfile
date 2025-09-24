# Use lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose MLflow UI port (optional)
EXPOSE 5000

# Default command (can be overridden)
ENTRYPOINT ["python", "train.py"]
