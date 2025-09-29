FROM python:3.11-slim

WORKDIR /app

# Install system deps (needed for some libs)
RUN apt-get update && apt-get install -y build-essential

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 8080

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
