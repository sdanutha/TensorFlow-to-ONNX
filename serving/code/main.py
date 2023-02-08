
# 
# uvicorn main:app --host=0.0.0.0 --port=8000 --reload
# 

import io
import time

from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi import Form
from fastapi.responses import JSONResponse

import onnxruntime as rt
import numpy as np

from PIL import Image

# Initial FastAPI
app = FastAPI(title = 'TensorFlow-to-ONNX-Serving')

# Load Model GPU
model_gpu = rt.InferenceSession(
    'model/model.onnx', providers = ['CUDAExecutionProvider']
)

# Load Model CPU
model_cpu = rt.InferenceSession(
    'model/model.onnx', providers = ['CPUExecutionProvider']
)

# Load Labels
with open('model/label.txt', 'r') as f:
    labels = f.read().split('\n')

# Information
@app.get('/')
async def information():
    return { 'title': 'TensorFlow-to-ONNX-Serving' }

# Inference
@app.post('/inference', tags = ['Inference'])
async def prediction(image: UploadFile = File(...), providers: str = Form(...)):
    
    # Get TimeStart
    start = time.time()
    
    # Set -> X
    x = Image.open(io.BytesIO(await image.read())).resize((224, 224))
    
    # Prep -> X
    x = np.array(x, dtype = np.float32)
    x = np.expand_dims(x, axis = 0)
    x = np.divide(x, 255.0)
    
    try:
        
        # Get -> Y
        if providers == 'GPU':
            y = model_gpu.run(['logits'], {'inputs': x})
        else:
            y = model_cpu.run(['logits'], {'inputs': x})
        
        # Outputs
        outputs = {
            'prediction': labels[y[0].argmax()].upper(),
            'time': (time.time() - start) * 1000,
            'providers': providers,
            'error': ''
        }
    
    except Exception as e:
        
        # Outputs
        outputs = { 'prediction': 'UNLNOW', 'time': -1, 'providers': 'UNLNOW', 'error': str(e) }
    
    finally:
        
        # Return
        return JSONResponse(outputs)
