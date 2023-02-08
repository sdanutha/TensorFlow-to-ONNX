
# 
# uvicorn main:app --host=0.0.0.0 --port=8000 --reload
# 

import io
import time

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi.responses import JSONResponse

import onnxruntime as rt
import numpy as np

from PIL import Image

# Initial FastAPI
app = FastAPI(title = 'TensorFlow-to-ONNX-Serving')


# load model
model = rt.InferenceSession(
    'model/model.onnx', providers = ['CUDAExecutionProvider']
)

# load labels
with open('model/label.txt', 'r') as f:
    labels = f.read().split('\n')


# Information
@app.get('/')
async def information():
    return { 'title': 'TensorFlow-to-ONNX-Serving' }

# Inference
@app.post('/inference', tags = ['Inference'])
async def prediction(image: UploadFile = File(...)):
    
    # timestart
    start = time.time()
    
    # set X
    x = await image.read()
    x = Image.open(io.BytesIO(x))
    
    # prep X
    x = x.resize((224, 224))
    x = np.array(x, dtype = np.float32)
    x = np.expand_dims(x, axis = 0)
    x = np.divide(x, 255.0)
    
    try:
        
        # get Y
        y = model.run(['logits'], {'inputs': x})
        
        # outputs
        outputs = {
            'prediction': labels[y[0].argmax()].upper(),
            'time': (time.time() - start) * 1000,
            'error': ''
        }
    
    except Exception as e:
        
        # outputs
        outputs = { 'prediction': 'UNLNOW', 'time': -1, 'error': str(e) }
    
    finally:
        
        # return
        return JSONResponse(outputs)
