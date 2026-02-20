FROM python:3.12.12-slim

WORKDIR /app

# Install system dependency for LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

EXPOSE 10000

CMD ["streamlit", "run", "src/ceras/streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
