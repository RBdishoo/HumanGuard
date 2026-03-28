FROM python:3.11-slim

WORKDIR /app

COPY requirements-prod.txt ./requirements-prod.txt
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

EXPOSE 8080
ENV PORT=8080

CMD ["python", "-m", "backend.app"]
