FROM ubuntu:20.04

MAINTAINER Piotr Szymanski "niedakh@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

WORKDIR /app

COPY . /app

RUN pip3 install -U flask spacy transformers

RUN python3 -m spacy download en_core_web_sm

RUN pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install flask-htpasswd sentencepiece

RUN python3 ./run_to_download_models.py

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]

