# AI Song Cover


## 1. Setup: 

```bash
git clone https://github.com/datdonq/SongCover
cd SongCover

pip install -r requirements.txt
python src/download_models.py
sh download_models.sh

# start gradio
python main.py
# start api
uvicorn app:app
```

## 2.  Notes
Current code only support some models on configs/RVC.json

## DEMO 

![image](https://github.com/datdonq/SongCover/assets/93565338/883a505e-01dd-4e36-94d3-3401d075ece9)
