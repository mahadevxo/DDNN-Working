# initialise remote machine
sudo apt update && sudo apt install vim && sudo apt install unzip && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash ./Miniconda3-latest-Linux-x86_64.sh && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda init && touch ~/.no_auto_tmux && exit
conda create -n ML python==3.10 && conda activate ML && pip install numpy torch torchvision matplotlib pandas tensorboardx tqdm scikit-image scikit-learn cma-es && git clone "https://github.com/mahadevxo/DDNN-Working.git"
cd DDNN-Working/src/pruning/taylor_series && curl -L -o ./imagenetmini-1000.zip https://www.kaggle.com/api/v1/datasets/download/ifigotin/imagenetmini-1000 sudo apt install unzip && unzip imagenetmini-1000.zip 
sudo apt install nvtop && conda deactivate && pip install nvitop && conda activate ML && pip install nvitop
cd ~/DDNN-Working/src/MVCNN && curl -L -o ./ModelNet40.zip http://modelnet.cs.princeton.edu/ModelNet40.zip && unzip ModelNet40.zip && rm ModelNet40.zip && python3 multi_convert.py

