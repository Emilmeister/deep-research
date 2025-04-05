FROM python:3.11

WORKDIR /app

COPY ./researcher .

RUN pip install -r requirements.txt

ENTRYPOINT ["sh", "-c", "nohup python main.py & tail -f nohup.out"]