# syntax=docker/dockerfile:1
FROM python:3.12.1-slim  # Smaller base image

# Install system dependencies
COPY system-packages.txt /tmp/system-packages.txt
RUN apt-get update && \
    xargs -a /tmp/system-packages.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set workdir and copy requirements first (for better caching)
WORKDIR /app
COPY requirements-global.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-global.txt

# Copy the rest of the app
COPY . .

# Streamlit settings
EXPOSE 8501
ENV PYTHONUNBUFFERED=1
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
