ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock \
    && useradd --create-home --uid 10001 vse

COPY evaluator /opt/vse/evaluator
COPY student/reference_baseline.py /opt/vse/student/reference_baseline.py
USER 10001:10001
WORKDIR /run

ENTRYPOINT ["python", "/opt/vse/evaluator/trusted_evaluator.py"]
