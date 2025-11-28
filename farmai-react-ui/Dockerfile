# Python image use karenge
FROM python:3.9-slim

WORKDIR /app

# Requirements file copy karo
COPY requirements.txt .

# Flask install karo
RUN pip install --no-cache-dir -r requirements.txt

# Baaki files copy karo
COPY dist ./dist
COPY main.py .

# Command to run
CMD ["python", "main.py"]