from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from TTS.api import TTS
from diffusers import StableDiffusionPipeline
from openai import OpenAI
import torch

app = FastAPI()

# Voice setup
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# OpenAI key (set this in Render.com as env var)
openai = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Image model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

@app.get("/")
def root():
    return {"message": "India.AI Backend is alive!"}

@app.post("/chat")
def chat(prompt: str = Query(...)):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}]
    )
    return JSONResponse({"reply": completion.choices[0].message.content})

@app.get("/speak")
def speak(text: str = Query(...)):
    tts_model.tts_to_file(text=text, file_path="speech.wav")
    return FileResponse("speech.wav", media_type="audio/wav")

@app.get("/image")
def image(prompt: str = Query(...)):
    image = pipe(prompt).images[0]
    image.save("output.png")
    return FileResponse("output.png", media_type="image/png")
