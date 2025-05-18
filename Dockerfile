FROM python:3.10-slim

# Disable oneDNN optimizations if desired
ENV TF_ENABLE_ONEDNN_OPTS=0

# Set working directory
WORKDIR /app

# Copy code and requirements
COPY app/ /app/app/
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI app port
EXPOSE 9000


# Run FastAPI using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]

