
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
app = FastAPI(
    title = 'TensorFlow-to-ONNX-Serving'
)

# load model
model = rt.InferenceSession(
    'model/model.onnx', providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
)

# load labels
with open('model/label.txt', 'r') as f:
    labels = f.read().split('\n')

# Inference API
@app.post('/inference', tags = ['Inference'])
async def prediction(image: UploadFile = File(...)):
    
    # set X
    x = await image.read()
    x = Image.open(io.BytesIO(x))
    
    x = np.array(x, dtype = np.float32)
    x = np.expand_dims(x, axis = 0)
    x = np.divide(x, 255.0)
    
    try:
        
        # get Y (Inference)
        start = time.time()
        y = model.run(['logits'], {'inputs': x})
        end = time.time()
        
        # outputs
        outputs = {
            'prediction': labels[y[0].argmax()].upper(),
            'time': (end - start) * 1000,
            'error': ''
        }
    
    except Exception as e:
        
        # outputs
        outputs = { 'prediction': 'UNLNOW', 'time': -1, 'error': str(e) }
    
    # return
    return JSONResponse(outputs)
