# Use an official Python base image
FROM python:3.9-slim

# Install MLflow
RUN pip install mlflow

# Expose port 5000
EXPOSE 5000

# Set entrypoint to run MLflow server
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
