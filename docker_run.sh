docker run --runtime=nvidia -it --rm -v --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:v1.0.0b1
# explain what above command does
# --runtime=nvidia: use nvidia runtime
# -it: interactive mode
# --rm: remove container after exit
# -v: mount volume
# --shm-size: shared memory size
# -p: port mapping
# --ulimit: set ulimit
# nvcr.io/nvidia/nemo:v1.0.0b1: docker image name
