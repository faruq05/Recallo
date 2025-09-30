# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential

# Install Node.js (for frontend build)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy backend & ai-engine requirements
COPY ai-engine/requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy backend and frontend
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Build React frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Move build files into Flask static folder
WORKDIR /app
RUN mkdir -p /app/backend/static && cp -r /app/frontend/dist/* /app/backend/static/

# Set working directory to backend
WORKDIR /app/backend

# Expose port
EXPOSE 5000

# Start with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
