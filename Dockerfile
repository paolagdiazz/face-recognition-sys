FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY facerec/ facerec/
COPY models/ models/
COPY data/ data/

# db/ y secrets/ se montan como volúmenes (ver docker-compose.yml)
RUN mkdir -p db secrets

ENTRYPOINT ["python", "-m", "facerec"]
CMD ["--help"]
