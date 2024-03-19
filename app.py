import uvicorn
from fastapi import FastAPI
import os
import json
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
from routers.v1.rvc import RVCModel
from src.mdx import load_mdx_model
from src.rvc import Config,load_hubert
from src.rmvpe import RMVPE
from main import run_cover,get_rvc_model
from fastapi import APIRouter, HTTPException, UploadFile, Form,Response
from fastapi.responses import FileResponse
from src.rvc import get_vc
from enum import Enum
app = FastAPI()

with open("mdxnet_models/model_data.json") as infile:
        mdx_model_params = json.load(infile)
device = 'cuda:0'
config = Config(device, True)
mdx_sess_1,model_1=load_mdx_model(mdx_model_params,"mdxnet_models/UVR-MDX-NET-Voc_FT.onnx")
mdx_sess_2,model_2=load_mdx_model(mdx_model_params,"mdxnet_models/UVR_MDXNET_KARA_2.onnx")
hubert_model = load_hubert(device, config.is_half, "rvc_models/hubert_base.pt")
model_rmvpe = RMVPE("rvc_models/rmvpe.pt", is_half=config.is_half, device=device)
CONFIG_FILE = "configs/RVC.json"
with open(CONFIG_FILE, "r") as file:
    User_config=json.load(file)
rvc_model=User_config["rvc_model"]
pitch_change=User_config["pitch_change"]
model_dict = {}
for name in rvc_model:
    rvc_model_path, rvc_index_path = get_rvc_model(name)
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)
    model=[cpt, version, net_g, tgt_sr, vc,rvc_index_path]
    model_dict[name] = model
class Pitch(Enum):
    NO_CHANGE = "no-change"
    MALE_TO_FEMALE = "male-to-female"
    FEMALE_TO_MALE = "female-to-male"
@app.post("/run_cover")
async def run_cover_endpoint(
    youtube_url: str = Form(...),
    voice_name: str = Form(...), 
    pitch_change: Pitch= Form(...)):
    await run_cover(youtube_url, voice_name, pitch_change, [mdx_sess_1, model_1],[mdx_sess_2, model_2], hubert_model, model_rmvpe,model_dict)
@app.get("/play_audio")
async def play_audio():
    url="song_output/result.mp3"
    return FileResponse(url, media_type="audio/mpeg")