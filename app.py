from fastapi import FastAPI,Form,HTTPException,Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow
from fastapi.responses import PlainTextResponse,HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Literal
import joblib as jb
import numpy as np

app=FastAPI()

templates=Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")

scaler=jb.load("scaler.pkl")
encoder=jb.load("encoder.pkl")
model=jb.load("model.pkl")

@app.get("/",response_class=HTMLResponse)
def about(request:Request):
    return templates.TemplateResponse("index.html",{"request":request,"title":"Preduct flower Species"})

@app.post("/predict",response_class=HTMLResponse)
def predict(
        request:Request,
        sepallength:float =Form(...,ge=0,description="length of sepal in cm"),
        sepalwidth:float =Form(...,ge=0,description="width of sepal in cm"),
        petalength:float =Form(...,ge=0,description="length of petal in cm"),
        petawidth:float =Form(...,ge=0,description="width of petal in cm")
):
    input=np.array([[sepallength,sepalwidth,petalength,petawidth]])
    scale_input=scaler.transform(input)
    output_prob=model.predict(scale_input)
    output=np.argmax(output_prob,axis=1)
    output=encoder.inverse_transform(output)
    return templates.TemplateResponse("index.html",{'request':request,"result":output[0]})

@app.put("/predict", response_class=PlainTextResponse)
def update_prediction(
    sepallength: float = Form(..., ge=0),
    sepalwidth:  float = Form(..., ge=0),
    petalength:  float = Form(..., ge=0),
    petawidth:   float = Form(..., ge=0)
):
    try:
        inp = np.array([[sepallength, sepalwidth, petalength, petawidth]])
        scaled = scaler.transform(inp)
        probs = model.predict(scaled)
        label = encoder.inverse_transform(np.argmax(probs, axis=1))[0]
        return label
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── DELETE: accept the same four fields, return confirmation ─────────────────
@app.delete("/predict", response_class=PlainTextResponse)
def delete_prediction(
    sepallength: float = Form(..., ge=0),  
    sepalwidth:  float = Form(..., ge=0),
    petalength:  float = Form(..., ge=0),
    petawidth:   float = Form(..., ge=0)
):
    # No persistent store to erase, so we simply acknowledge the request
    # (Optionally, echo which flower *would* have been predicted).
    inp = np.array([[sepallength, sepalwidth, petalength, petawidth]])
    scaled = scaler.transform(inp)
    probs = model.predict(scaled)
    label = encoder.inverse_transform(np.argmax(probs, axis=1))[0]
    return f"deleted flower : {label}"


'''basically post use to update input for new prediction we assign new input for specific output and delete is used to assign the input to not give this output'''