# Use stable Python version
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose port (Render uses 10000)
EXPOSE 10000

# Start app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]