import ffmpeg
import numpy as np
import librosa

def load_audio(file, sr):
    # # try:
    #     # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
    #     # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
    #     # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
    #     file = (
    #         file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    #     )  # 防止小白拷路径头尾带了空格和"和回车
    #     print(file)
    #     out, _ = (
    #         ffmpeg.input(file, threads=0)
    #         .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
    #         .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    #     )
    # # except Exception as e:
    # #     raise RuntimeError(f"Failed to load audio: {e}")

    #     return np.frombuffer(out, np.float32).flatten()
    waveform, _ = librosa.load(file, sr=sr, mono=True)

    return waveform
# load_audio("/home/user/real-voice-clone-api/song_output/main_vocals_dereverb_path.wav",16000)