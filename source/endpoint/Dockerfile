ARG REGISTRY_URI
FROM ${REGISTRY_URI}/pytorch-inference:1.5.0-gpu-py36-cu101-ubuntu16.04
#FROM ${REGISTRY_URI}/pytorch-inference:1.8.1-gpu-py36-cu111-ubuntu18.04

RUN mkdir -p /opt/ml/model

# COPY package/ /opt/ml/code/package/

# COPY serve.py /opt/ml/model/code/

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install networkx==2.3 flask gevent gunicorn boto3 opencv-python==4.5.1.48 matplotlib -i https://opentuna.cn/pypi/web/simple

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# Set up the program in the image
COPY * /opt/program/
COPY utils/* /opt/program/utils/
WORKDIR /opt/program

ENTRYPOINT ["python", "serve.py"]

