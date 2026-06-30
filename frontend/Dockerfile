FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    API_BASE_URL=http://backend:5000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend
COPY backend ./backend
COPY database ./database
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "frontend/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
