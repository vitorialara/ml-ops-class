FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn scikit-learn pandas joblib numpy

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]