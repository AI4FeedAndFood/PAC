from predict_estimates import generate, from_config_load_model
from unsloth import FastLanguageModel
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from constant import INSTRUCTION
import argparse

app = FastAPI()

class TextGenerationRequest(BaseModel):
    text_ocr: str

class TextGenerationResponse(BaseModel):
    prediction: str

@app.post("/generate_text/", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    global model, tokenizer
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model and/or tokenizer not loaded")

    try:
        pred = generate(INSTRUCTION, request.text_ocr, model, tokenizer, echo=False)
        return TextGenerationResponse(prediction=pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_model_and_tokenizer(config_path):
    global model, tokenizer
    model, tokenizer = from_config_load_model(config_path)

if __name__ == "__main__":
    
    # Remplacez les valeurs par défaut par les vôtres
    parser = argparse.ArgumentParser(description="Lunch server")
    parser.add_argument("path_config", type=str, help="Path for config_predict.json, ex: 'what/is/the/directory/config_predict.json'")
    args = parser.parse_args()
    load_model_and_tokenizer(args.path_config)
    uvicorn.run(app, host="127.0.0.1", port=8000)