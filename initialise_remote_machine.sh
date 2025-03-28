# initialise remote machine
    sudo apt update && sudo apt install vim && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash ~/Miniconda3-latest-Linux-x86_64.sh && conda init && exit
conda create -n ML python==3.10 && conda activate ML && pip install numpy torch torchvision matplotlib pandas && git clone "https://github.com/mahadevxo/DDNN-Working.git"