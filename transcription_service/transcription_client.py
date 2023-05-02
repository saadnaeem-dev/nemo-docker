import grpc
import transcription_service_pb2
import transcription_service_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = transcription_service_pb2_grpc.TranscriptionServiceStub(channel)

    model_path = r"C:\\Users\\saadn\\PycharmProjects\\Nvidia-Nemo-Models\\stt_en_squeezeformer_ctc_xsmall_ls.nemo"
    model_name = "stt_en_squeezeformer_ctc_xsmall_ls.json"
    audio_directory_path = r"C:\Users\saadn\PycharmProjects\DATA\295000_wav_files"

    response = stub.TranscribeAudio(
        transcription_service_pb2.TranscriptionRequest(
            model_path=model_path,
            audio_directory_path=audio_directory_path,
            model_name=model_name,
        )
    )

    print("Server response:")
    print(f"Model path: {response.model_path}")
    print(f"Audio directory path: {response.audio_directory_path}")

if __name__ == '__main__':
    run()
