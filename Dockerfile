FROM python:3.10-slim-bullseye

WORKDIR /main
 
COPY . /main

COPY ./requirements.txt /app/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
 
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8001" , "main:app"]

# CMD ["python", "main.py", "--host=0.0.0.0"]