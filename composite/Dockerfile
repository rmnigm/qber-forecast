FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

RUN pip install --no-cache-dir --upgrade pip

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080/tcp

CMD ["python", "main.py"]
