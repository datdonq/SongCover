import torch
import sys
import os
sys.path.insert(0, os.path.abspath("src"))
from mdx import run_mdx
import librosa
import numpy as np
import json
from mdx import load_mdx_model
import soundfile as sf
from rvc import Config, load_hubert, get_vc, rvc_infer
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
from scipy.io.wavfile import write
import yt_dlp
import gradio as gr
import io
from rmvpe import RMVPE
def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    # check if mono
    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path

def get_rvc_model(voice_model_name):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join("rvc_models", voice_model_name)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''
def voice_change(voice_model_name, vocals_path, output_path, pitch_change,model_rmvpe,hubert_model,model_dict):
    # device = 'cuda:0'
    # config = Config(device, True)
    # rvc_model_path, rvc_index_path = get_rvc_model(voice_model_name)
    cpt, version, net_g, tgt_sr, vc, rvc_index_path = model_dict[voice_model_name]
    index_rate=0.5
    f0_method='rmvpe'
    filter_radius=3
    rms_mix_rate=0.25
    protect=0.33
    crepe_hop_length=128
    rvc_infer(model_rmvpe,rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
def seperate(orig_song_path,mdx1,mdx2):   
    orig_song_path = convert_to_stereo(orig_song_path)
    vocals_path, instrumentals_path,sr = run_mdx(mdx1, orig_song_path)
    sf.write("song_output/vocals_path.wav", vocals_path, sr)
    sf.write("song_output/instrumentals_path.wav", instrumentals_path, sr)
    
    backup_vocals_path, main_vocals_path,sr = run_mdx(mdx2,"song_output/vocals_path.wav" )
    sf.write("song_output/backup_vocals_path.wav", backup_vocals_path, sr)
    sf.write("song_output/main_vocals_path.wav", main_vocals_path, sr)
    
    # _, main_vocals_dereverb_path,sr = run_mdx(mdx_sess_3,model_3,"song_output/main_vocals_path.wav" )
    # sf.write("song_output/main_vocals_dereverb_path.wav", main_vocals_dereverb_path, sr)
def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping):
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    # Initialize audio effects plugins
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
         ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path
def combine_audio(audio_paths, output_path, main_gain, backup_gain, inst_gain, output_format):
    # Đọc các tệp âm thanh và áp dụng điều chỉnh âm lượng
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format=output_format)
def yt_download(url):
    AUDIO_NAME = "test"
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "outtmpl": f"song_input/{AUDIO_NAME}",  # this is where you can edit how you'd like the filenames to be formatted
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
async def run_cover(youtube_url, voice_name,pitch_change,mdx1,mdx2,hubert_model,model_rmvpe,model_dict):
    
    # # Step 1: Download song from youtube (assuming yt_download is asynchronous)
    yt_download(youtube_url)

    # # Step 2: Separate song into patches
    seperate("song_input/test.wav",mdx1,mdx2)  # No change needed for synchronous functions

    # Step 3: Change main voice into AI voice (assuming voice_change is asynchronous)
    if pitch_change =="no-change":
        pitch=0
    elif pitch_change =="male-to-female":
        pitch=1
    else:
        pitch=-1
    voice_change(voice_name, "song_output/main_vocals_path.wav", "song_output/change.wav", pitch,model_rmvpe,hubert_model,model_dict)

    # Add audio effects (assuming add_audio_effects is asynchronous)
    add_audio_effects("song_output/change.wav", reverb_rm_size=0.15, reverb_wet=0.2, reverb_dry=0.8, reverb_damping=0.7)

    # Step 4: Combine all and save (assuming combine_audio is asynchronous)
    combine_audio(["song_output/change_mixed.wav","song_output/backup_vocals_path.wav", "song_output/instrumentals_path.wav"], "song_output/result.mp3", main_gain=0, backup_gain=0, inst_gain=0, output_format="mp3")

    # Return the path to the generated audio file
    return "song_output/result.mp3"
async def run_gradio(youtube_url, voice_name,pitch_change):
        await run_cover(youtube_url, voice_name,pitch_change,[mdx_sess_1,model_1],[mdx_sess_2,model_2],hubert_model,model_rmvpe,model_dict)
        return "song_output/result.mp3"
if __name__=="__main__":
    with open("mdxnet_models/model_data.json") as infile:
        mdx_model_params = json.load(infile)
    device = 'cuda:0'
    config = Config(device, True)
    mdx_sess_1,model_1=load_mdx_model(mdx_model_params,"mdxnet_models/UVR-MDX-NET-Voc_FT.onnx")
    mdx_sess_2,model_2=load_mdx_model(mdx_model_params,"mdxnet_models/UVR_MDXNET_KARA_2.onnx")
    # mdx_sess_3,model_3=load_mdx_model(mdx_model_params,"mdxnet_models/Reverb_HQ_By_FoxJoy.onnx")
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
        model=[cpt, version, net_g, tgt_sr, vc, rvc_index_path]
        model_dict[name] = model
    gr.Interface(
    fn=run_gradio,
    inputs=[gr.inputs.Textbox(label="URL"), gr.inputs.Dropdown(label="AI_Name",choices=rvc_model),gr.inputs.Dropdown(label="Choose pitch_change", choices=pitch_change)],
    outputs=gr.outputs.Audio(label="File audio", type='filepath'),
    title="AI Song Cover",
    description="Enter URL and choose voice name."
    ).launch(share=True)
    