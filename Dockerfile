FROM ubuntu:20.04

MAINTAINER Piotr Szymanski "niedakh@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

WORKDIR /app

COPY . /app

RUN pip flask spacy transformers

RUN python ./run_to_download_models.py

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]

