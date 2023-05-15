# import subprocess
#
# def wav_to_g711(input_file, output_file):
#     subprocess.run(['ffmpeg', '-i', input_file, '-c:a', 'pcm_alaw', '-ar', '8000', '-ac', '1', output_file])  # Use 'pcm_mulaw' for G.711 u-law
#
# def g711_to_wav(input_file, output_file):
#     subprocess.run(['ffmpeg', '-f', 'alaw', '-ar', '8000', '-ac', '1', '-i', input_file, '-c:a', 'pcm_s16le', '-ar', '8000', '-ac', '1', output_file])  # Use '-f', 'mulaw' for G.711 u-law
#
# # Usage example
# wav_to_g711("concatenated_audio.wav", "output.g711")
# g711_to_wav("output.g711", "output_converted.wav")
import subprocess

def wav_to_g711(input_file, output_file):
    subprocess.run(['ffmpeg', '-i', input_file, '-c:a', 'pcm_alaw', '-ar', '8000', '-ac', '1', '-f', 'alaw', output_file])  # Use 'pcm_mulaw' for G.711 u-law

def g711_to_wav(input_file, output_file):
    subprocess.run(['ffmpeg', '-f', 'alaw', '-ar', '8000', '-ac', '1', '-i', input_file, '-c:a', 'pcm_s16le', '-ar', '8000', '-ac', '1', output_file])  # Use '-f', 'mulaw' for G.711 u-law

# Usage example
wav_to_g711("concatenated_audio.wav", "output.g711")
g711_to_wav("output.g711", "output_converted.wav")
