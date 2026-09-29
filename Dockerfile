# Driver image: Python + Java + PySpark (local mode, or submit to the compose cluster)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 PYTHONUNBUFFERED=1 TZ=UTC
WORKDIR /opt/project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline ./pipeline
CMD ["python", "-m", "pipeline.run", "--lake", "/data/lake", "--rows", "2000000"]
