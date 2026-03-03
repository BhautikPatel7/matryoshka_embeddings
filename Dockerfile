FROM python:3.11-slim
WORKDIR /app
COPY requirenments.txt .
RUN pip install --no-cache-dir -r requirenments.txt
COPY . .
CMD [ "python", "main.py"]