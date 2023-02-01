
# base images
FROM python:3.7-slim

# source code
COPY ./code /code
WORKDIR /code

# install dependencies
RUN pip install -r requirements.txt

# run
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
