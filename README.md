## Oskar Walawender, lab 4

Repozytorium: 

### Instrukcja uruchomienia

#### Lokalnie:
1. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   ```
2. Uruchom aplikację:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
3. Wysyłaj zapytania POST na endpoint `/predict` z danymi cech irysa.

#### Za pomocą Dockera:
1. Zbuduj obraz Dockera:
   ```bash
   docker build -t iris-classifier .
   ```
2. Uruchom kontener:
   ```bash
   docker run -p 8000:8000 iris-classifier
   ```

#### Za pomocą Docker Compose:
1. Uruchom aplikację:
   ```bash
   docker-compose up
   ```
2. Aplikacja będzie dostępna pod adresem `http://127.0.0.1:8000`.

### Testowanie API
Przykład zapytania POST:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
```


