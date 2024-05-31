import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from constant import INSTRUCTION

app = FastAPI()

class TextGenerationRequest(BaseModel):
    text_ocr: str

class TextGenerationResponse(BaseModel):
    prediction: str

@app.post("/generate_text/", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    global factor
    if not factor :
        raise HTTPException(status_code=500, detail="Factor not loaded")

    try:
        pred = generate(request.text_ocr, factor)
        return TextGenerationResponse(prediction=pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_model_and_tokenizer():
    global factor
    factor = 2
def generate(ocr_text, factor):
    return factor * ocr_text

if __name__ == "__main__":
    # Remplacez les valeurs par défaut par les vôtres
    load_model_and_tokenizer()
    uvicorn.run(app, host="127.0.0.1", port=8000)