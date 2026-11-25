FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt requirements-image.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-image.txt
COPY app app
COPY models models
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
