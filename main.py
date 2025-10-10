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
import torchvision
import database

app = FastAPI()

# trained model for inference
model = skin_cnn()
model.load_state_dict(torch.load("saved_models/skin_model_epoch_7.pth"))
to_tensor = torchvision.transforms.ToTensor()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def analyze_image(body):
    img = body["image"]
    image_bytes = base64.b64decode(img)     # string -> bytefile
    image_file = Image.open(io.BytesIO(image_bytes)) # bytefile -> imagefile -> open the image
    image_file = image_file.rotate(270)              # rotate to proper dimensions
    image_file = Image.open("cancerous2.jpg")      # use for testoing

    print("passing the image file to the model " + str(image_file))
    result = model(image_file)
    print("the inference on the provided user image is : " + str(result[0][0]))

    # return a JSON object to the user
    if result[0][0] > 0.5:
        return {"result" : True}
    else:
        return  {"result" : False}
    


    

# Add a root endpoint that accepts all different structures of JSON
# one key : sending an image
# two keys : logging into a profile
# three keys : making a new profile
@app.post("/")
async def root_post(request: Request):
    database.init_db()      # initialize the database
    body = await request.json()
    if len(body.keys()) == 1: 
        return analyze_image(body)
    elif len(body.keys()) == 2:
        return {"result" :  database.login(body["username"], body["password"])}
    else:
        print("creating new user")
        return {"result" : database.create_user(body["username"], body["password"], body["repeat_password"])}



    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)