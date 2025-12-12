import torch
import io

# This adds type hints and checking to our data 
from pydantic import BaseModel 

from torchvision.models import ResNet

from fastapi import FastAPI, File, UploadFile, Depends

from app.model import load_model, load_transforms 

from torchvision.transforms import v2 as transforms

from PIL import Image

import torch.nn.functional as F

CATEGORIES = ['freshapple', 'freshbanana',
              'freshorange', 'rottenapple', 
              'rottenbanana', 'rottenorange']


# This is a data model that describes the output of the API
class Result(BaseModel):
    category: str
    confidence: float
    

# Create the FastAPI instance
app = FastAPI() 

# Debug message to check that the app is running
@app.get('/')
def read_root():
    return {"message": "API is running. Visit /docs for the Swagger API documentation"}


@app.post('/predict', response_model=Result)
async def predict(
    input_image: UploadFile = File(...),
    model: ResNet = Depends(load_model),
    transforms: transforms.Compose = Depends(load_transforms)
    
) -> Result: 
    image = Image.open(io.BytesIO(await input_image.read()))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Add the batch dimension for inference
    image = transforms(image).reshape(1, 3, 224, 224)
    
    # Turn off Dropout and BatchNorm uses stats from training
    model.eval()
    with torch.inference_mode():
        # We turn off gradient tracking for inference
        # this saves memory and makes inference faster
        output = model(image)
        category = CATEGORIES[output.argmax()]
        confidence = F.softmax(output, dim=1).max().item()
        return Result(category=category, confidence=confidence)