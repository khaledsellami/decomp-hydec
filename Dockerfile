FROM python:3.9-slim

RUN mkdir /service
COPY protos/ /protos/
WORKDIR /service
COPY . .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

EXPOSE 50055
ENTRYPOINT [ "python", "server.py" ]