conda create --name nemo python==3.8.10
conda activate nemo
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pip
pip install Cython
pip install nemo_toolkit['all']
pip install grpcio grpcio-tools
