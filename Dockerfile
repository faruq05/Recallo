# ========================
# Stage 1: Build frontend
# ========================
FROM node:20-bullseye AS frontend-builder

# Set working directory
WORKDIR /app/frontend

# Copy frontend package files and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy full frontend code and build
COPY frontend/ ./
RUN npm run build

# ========================
# Stage 2: Build backend + final image
# ========================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Python & Node tools if needed
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY ai-engine/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy backend code
COPY backend/ ./backend/

# Copy frontend build from previous stage
COPY --from=frontend-builder /app/frontend/dist ./backend/static

# Set working directory to backend
WORKDIR /app/backend

# Expose backend port
EXPOSE 5000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
