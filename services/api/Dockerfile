FROM python:3.12-slim
WORKDIR /app
COPY services/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
 # the OCR package pulls in desktop OpenCV, which needs X11 libraries a server doesn't have; keep the headless build
 && pip uninstall -y opencv-python \
 && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless \
 && python -c "import cv2, rapidocr_onnxruntime, pymupdf"
COPY services/api /app
COPY apps/web /web
ENV GV_WEB_DIR=/web GV_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "gv.api:app", "--host", "0.0.0.0", "--port", "8000"]
