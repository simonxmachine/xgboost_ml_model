For loading into AWS Fargate and API Gateway

1) Ensure CORS is enabled with flask-cors library, and run on gunicorn
2) Dockerize image and push to ECR
3) Create new ECS cluster and task that connects to the URI endpoint of the Fargate container
4) Connect AWS API Gateway to task public URL
5) Configure CORS in API Gateway to allow * for origin, and include methods Option, Post, Head, *
6) Add routes to the extension /xg_predict
7) Ensure the proper domain URI is assigned to the correct request type



*** 

Need use Python version 3.10 to use tensorflow. 

To set up virtual environment: python3.10 -m venv env

If having issue with Tensorflow install:
pip3 install --no-cache-dir  --force-reinstall -Iv tensorflow
