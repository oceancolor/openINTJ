FROM python:3.11-slim

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HUNYUAN_BASE_URL="https://api.hunyuan.cloud.tencent.com/v1"
ENV HUNYUAN_MODEL="hunyuan-turbos-latest"
ENV HUNYUAN_VISION_MODEL="hunyuan-vision"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]