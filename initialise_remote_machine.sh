# initialise remote machine
bash -e -c 'sudo apt update ; sudo apt install -y vim unzip ; wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ; bash ./Miniconda3-latest-Linux-x86_64.sh ; eval "$(/root/miniconda3/bin/conda shell.bash hook)" ; conda init ; touch ~/.no_auto_tmux ; exit'

bash -e -c 'conda create -n ML python==3.10 ; conda activate ML ; pip install numpy torch torchvision matplotlib pandas tensorboardx tqdm scikit-image scikit-learn cma-es ; git clone "https://github.com/mahadevxo/DDNN-Working.git" ; sudo apt install -y nvtop ; conda deactivate ; pip install nvitop ; conda activate ML ; pip install nvitop'

bash -e -c 'cd DDNN-Working/src/test-models/ ; curl -L -o ./imagenetmini-1000.zip https://www.kaggle.com/api/v1/datasets/download/ifigotin/imagenetmini-1000 ; sudo apt install -y unzip ; unzip imagenetmini-1000.zip'

bash -e -c 'cd ~/DDNN-Working/src/MVCNN ; curl -L -o ./ModelNet40.zip http://modelnet.cs.princeton.edu/ModelNet40.zip ; unzip ModelNet40.zip ; rm ModelNet40.zip ; python3 multi_convert.py'