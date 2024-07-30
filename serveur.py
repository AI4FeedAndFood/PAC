# server.py
"""
This script sets up a FastAPI server for Estimates predictions.
It loads a pre-trained model and tokenizer, and provides an endpoint for Estimates predictions.
"""

from predict_estimates import generate, from_config_load_model, split_into_list
from unsloth import FastLanguageModel
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from constant import INSTRUCTION
import argparse
import time

# Initialize FastAPI application
app = FastAPI()

# Define request and response models
class EstimatesRequest(BaseModel):
    text_ocr: str

class EstimatesResponse(BaseModel):
    prediction: list

# Define global variables for model and tokenizer
model = None
tokenizer = None
model_name = None

@app.post("/predict_estimates/", response_model=EstimatesResponse)
async def generate_text(request: EstimatesRequest):
    """
    Endpoint for text generation predictions.
    
    Args:
        request (TextGenerationRequest): The request containing OCR text.
    
    Returns:
        TextGenerationResponse: The response containing the prediction.
    
    Raises:
        HTTPException: If the model or tokenizer is not loaded, or if an error occurs during prediction.
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model and/or tokenizer not loaded")

    try:
        start_time = time.time()
        
        # Generate prediction
        pred = generate(INSTRUCTION, request.text_ocr, model, tokenizer, echo=False)
        list_pred = split_into_list(pred)   
        print(f"Inference takes : {time.time() - start_time:.0f}s")
        
        return EstimatesResponse(prediction=list_pred)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_model_and_tokenizer(config_path):
    """
    Load the model and tokenizer from the given configuration path.
    
    Args:
        config_path (str): Path to the configuration file.
    """
    global model, tokenizer, model_name
    model, tokenizer = from_config_load_model(config_path)
    FastLanguageModel.for_inference(model)
    model_name = model.config.name_or_path

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Launch server")
    parser.add_argument("path_config", type=str, help="Path for config_predict.json, ex: 'what/is/the/directory/config_predict.json'")
    args = parser.parse_args()
    
    # Load model and tokenizer
    load_model_and_tokenizer(args.path_config)
    
    # Run the FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8000)