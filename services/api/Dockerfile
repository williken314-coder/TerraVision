FROM python:3.12-slim
WORKDIR /app
COPY services/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/api /app
COPY apps/web /web
ENV GV_WEB_DIR=/web GV_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "gv.api:app", "--host", "0.0.0.0", "--port", "8000"]
