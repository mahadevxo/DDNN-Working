# initialise remote machine
sudo apt update && sudo apt install vim && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash ~/Miniconda3-latest-Linux-x86_64.sh
conda init && exit
conda create -n ML python==3.10 && conda activate ML && pip install numpy torch torchvision matplotlib && git clone "https://github.com/mahadevxo/DDNN-Working.git" && cd DDNN-Working/src/pruning/taylor_series && curl -L -o ./imagenetmini-1000.zip https://www.kaggle.com/api/v1/datasets/download/ifigotin/imagenetmini-1000 && sudo apt install unzip && unzip imagenetmini-1000.zip
echo "Done"