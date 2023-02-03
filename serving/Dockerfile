
# base
FROM mcr.microsoft.com/azureml/onnxruntime

# code
COPY ./code /code
WORKDIR /code

# packages
RUN pip3 install --no-cache-dir -r requirements.txt

# run
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
