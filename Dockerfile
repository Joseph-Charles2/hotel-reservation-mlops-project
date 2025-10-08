# Create a python image
FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE =1 \
    PYTHONUNBUFFERED =1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

# Change your pip install line
RUN pip install --no-cache-dir pyarrow -e .

RUN python pipeline/training_pipeline.py

EXPOSE 5000

CMD ["python" , "application.py" ]