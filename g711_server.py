from flask import Flask, send_file
import subprocess
app = Flask(__name__)

@app.route('/stream.g711')
def stream_g711():
    print("Serving audio.g711...")
    return send_file('output.g711', mimetype='audio/g711')

if __name__ == '__main__':
    # if server already running on port 5000, kill it
    subprocess.run(["fuser", "-k", "5000/tcp"])
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)

# ffmpeg -i tutorials/datasets/mini-dev-clean/concatenated_audio.wav -acodec pcm_mulaw -ar 8000 -f mulaw audio.g711
# ffmpeg -f alaw -ar 8000 -i /path/to/your/file.g711 /path/to/your/output/file.wav