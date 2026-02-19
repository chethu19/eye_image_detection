# Use Python 3.11 to match training environment
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies if any (none strictly needed for basic flask/sklearn, but sometimes libgomp is needed for sklearn)
# generic helpful ones:
# RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Flask port (HF Spaces defaults to 7860)
EXPOSE 7860

# Start the application
CMD ["python", "app.py"]
