from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image
import base64
import io
from model import skin_cnn
import json

app = FastAPI()
model = skin_cnn()
model.load_state_dict(torch.load("saved_models/skin_model_epoch_7.pth"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserData(BaseModel):
    username: str
    password: str

class UserImage(BaseModel):
    image_string: str

# Add a root endpoint that accepts GET and POST
@app.post("/")
async def root_post(request: Request):
    body = await request.json()
    if len(body.keys()) == 2:
        return {"result" : True}
    else:
        print("loading the provided image")
        img = body["image"]
        image_bytes = base64.b64decode(img)     # string -> bytefile
        image_file = Image.open(io.BytesIO(image_bytes)) # bytefile -> imagefile -> open the image
        image_file = image_file.rotate(270)              # rotate to proper dimensions

        # provide inference from user image
        result = model(image_file)
        print("the inference on the provided user image is cancerous : " + str(result[0][0]))

        # return a JSON object to the user
        if result[0][0] > 0.5:
            return {"result" : "Cancerous"}
        else:
            return  {"result" : "Benign"}

    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)