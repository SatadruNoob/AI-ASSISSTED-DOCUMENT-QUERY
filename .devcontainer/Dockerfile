# syntax=docker/dockerfile:1
FROM python:3.12.1

# Install system dependencies
COPY system-packages.txt /tmp/system-packages.txt
RUN apt-get update && \
    xargs -a /tmp/system-packages.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY .. /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-global.txt

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables (optional, for Streamlit)
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["streamlit", "run", "app.py"]
