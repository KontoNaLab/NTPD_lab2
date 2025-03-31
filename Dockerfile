FROM python:3.12-slim

WORKDIR /app

COPY . /app

# Kopiowanie pliku requirements.txt i instalacja zależności
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
