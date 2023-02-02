
# base images
FROM mcr.microsoft.com/azureml/onnxruntime

# working directory
WORKDIR /code

# install packages
COPY ./code/requirements.txt /code
RUN pip3 install --no-cache-dir -r requirements.txt

# source code
COPY ./code /code

# run
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
