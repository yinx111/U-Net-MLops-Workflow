FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements_app.txt ./

RUN python -m pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2 && \
    pip install -r requirements_app.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["python", "-m", "app.app"]
