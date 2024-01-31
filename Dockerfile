FROM python:3.12-slim-bullseye

WORKDIR /main
 
COPY . /main

COPY ./requirements.txt /app/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
 
# CMD ["gunicorn", "-b", "0.0.0.0:8001" ,"main:main"]

CMD ["python", "main.py", "--host=0.0.0.0"]