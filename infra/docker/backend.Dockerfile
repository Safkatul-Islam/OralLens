FROM python:3.13-slim-bookworm

ARG TORCH_VERSION=2.12.1
ARG TORCHVISION_VERSION=0.27.1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/orallens/backend:/opt/orallens/ml/src

WORKDIR /opt/orallens

COPY infra/docker/backend-cpu-requirements.txt /tmp/backend-cpu-requirements.txt
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}+cpu" \
        "torchvision==${TORCHVISION_VERSION}+cpu" \
    && python -m pip install --no-cache-dir \
        --requirement /tmp/backend-cpu-requirements.txt \
    && rm /tmp/backend-cpu-requirements.txt

COPY backend/app /opt/orallens/backend/app
COPY ml/src /opt/orallens/ml/src
COPY ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml \
    /opt/orallens/ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml
COPY infra/render /opt/orallens/infra/render

RUN groupadd --gid 10001 orallens \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin orallens \
    && mkdir -p \
        /opt/orallens/ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug \
        /tmp/orallens/uploads \
        /tmp/orallens/artifacts \
    && sed -i 's/\r$//' /opt/orallens/infra/render/start-backend.sh \
    && chmod 0555 /opt/orallens/infra/render/start-backend.sh \
    && chown -R orallens:orallens \
        /opt/orallens/ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug \
        /tmp/orallens

USER 10001:10001

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "/opt/orallens/backend", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
