FROM python:3.11-slim

WORKDIR /

COPY requirements.txt requirements.txt
COPY server.sh server.sh

ENV GOOGLE_APPLICATION_CREDENTIALS="/secrets/sa-key.json"

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080

RUN chmod +x server.sh

ENTRYPOINT ["./server.sh"]