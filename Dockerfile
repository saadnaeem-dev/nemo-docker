FROM nvcr.io/nvidia/nemo:v1.0.0b1
ARG PROJECT_NAME
ARG PROJECT_NAMESPACE
WORKDIR ./$PROJECT_NAME
COPY . ./$PROJECT_NAME

# Expose port
EXPOSE 50051

# Start gRPC server and client
CMD ["python", "transcription_service/transcription_server.py"]

# Docker build command
# docker build -t local-nemo-docker-img .

# docker run -p 50051:50051 local-nemo-docker-img

# docker exec local-nemo-docker-img python "transcription_service/transcription_client.py" --arg1=value1 --arg2=value2