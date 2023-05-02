import grpc
from concurrent import futures
import transcription_service_pb2
import transcription_service_pb2_grpc
import os
class TranscriptionService(transcription_service_pb2_grpc.TranscriptionServiceServicer):
    def TranscribeAudio(self, request, context):
        model_path = request.model_path
        audio_directory_path = request.audio_directory_path
        os.system(r"python transcribe.py")
        print(f"Model path: {model_path}")
        print(f"Audio directory path: {audio_directory_path}")

        return transcription_service_pb2.TranscriptionResponse(
            model_path=model_path,
            audio_directory_path=audio_directory_path
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    transcription_service_pb2_grpc.add_TranscriptionServiceServicer_to_server(TranscriptionService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Transcription server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
