import io
import requests
import subprocess
import numpy as np
import torch
import whisper
from utils import ChunkBufferDecoder
from utils import correct_last_word


# Constants
CHUNK_SIZE = 4096
stride = 4
STREAM_URL = "http://localhost:5000/stream.g711"
MODEL_PATH = r"../../PycharmProjects/Nvidia-Nemo-Models/stt_en_squeezeformer_ctc_xsmall_ls.nemo"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the ASR model
model = whisper.load_model("base.en")
model = model.to(DEVICE)
SAMPLE_RATE  = 16000
model.eval()

# Function to convert G.711 audio chunks to PCM
def decode_g711_to_pcm(g711_data, chunk_size):
    ffmpeg_command = [
        "ffmpeg",
        "-f", "alaw",
        "-ar", "8000",
        "-ac", "1",
        "-i", "pipe:",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        "pipe:"
    ]

    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.stdin.write(g711_data)
    process.stdin.flush()
    process.stdin.close()

    while True:
        pcm_data = process.stdout.read(chunk_size)
        if not pcm_data:
            break
        yield pcm_data

    process.stdout.close()






# Read and transcribe streaming G.711 audio
def transcribe_streaming_g711(model, stride, chunk_len_in_secs=1, buffer_len_in_secs=5):
    """

    :param model: model to be used for transcribing
    :param stride: a token is produced for every 4 feature vectors in the time domain.  MelSpectrogram features are generated once every 10 ms, so a token is produced for every 40 ms of audio.
    :param chunk_len_in_secs: used to determine the number of samples to be sent to the model (control length of transcribed text)
    :param buffer_len_in_secs: used to determine the number of samples to be accumulated before sending to the model (control latency)
    :return:
    """
    print("Started reading G.711 stream...")

    accumulated_audio = np.array([], dtype=np.float32)
    accumulated_samples = int(buffer_len_in_secs * SAMPLE_RATE)

    with requests.get(STREAM_URL, stream=True) as response:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            for pcm_data in decode_g711_to_pcm(chunk, CHUNK_SIZE):
                pcm_audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / (2 ** 15)
                accumulated_audio = np.concatenate((accumulated_audio, pcm_audio))
                # what are accumulated_samples?
                # accumulated_samples is the total number of samples that the function will accumulate before feeding it into the model for transcription.
                if accumulated_audio.shape[0] >= accumulated_samples:
                    # what are buffers?
                    # buffers is a list of audio data chunks that the model will transcribe at a time.
                    buffers = [accumulated_audio]
                    transcription = model.transcribe(buffers[0])
                    print(f"Whisper Transcription: {transcription['text']}")
                    # print(f"Custom Merge Text + Corrected Transcription: {correct_last_word(transcription[1])}")
                    accumulated_audio = accumulated_audio[accumulated_samples:]





transcribe_streaming_g711(model, stride=4)
