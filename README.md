# Diabetes Predictor API with FastAPI and Docker

This is a FastAPI application that serves a machine learning model to predict diabetes. The app is containerized using Docker for easy deployment.

## Features

- FastAPI REST API to accept input features and return predictions.
- Interactive Swagger UI documentation at `/docs`.
- Dockerized for consistent and simple deployment.
- Example data validation and Pydantic models.

## Getting Started

### Prerequisites

- Docker installed: https://docs.docker.com/get-docker/
- (Optional) Python 3.11+ and virtual environment for local development.

### Running with Docker

1. Build the Docker image (run in project root where `Dockerfile` exists):

   ```bash
   docker build -t diabetes-predictor .
