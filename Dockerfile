FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt gunicorn

# Copy project
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Run the application with Gunicorn
# Replace `app:app` if your Flask instance has a different name
CMD ["gunicorn", "-w", "2", "-k", "sync", "-t", "600", "-b", "0.0.0.0:5000", "app:app"]
