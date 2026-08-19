ARG BASE_IMAGE=registry.intsig.net/zhuochu_yang/diffsynth@sha256:a38ddf970b5316d67fc453da1c0ef925484895797d2f2dd665369874a7f54333
FROM ${BASE_IMAGE}

RUN python -m pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    bitsandbytes==0.45.2

WORKDIR /opt/vse-smoke
COPY scripts/train_and_verify.py /opt/vse-smoke/train_and_verify.py

ENTRYPOINT ["python", "/opt/vse-smoke/train_and_verify.py"]
