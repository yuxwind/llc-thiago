conda create -n lossless python=3.6.5
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
pip install -r dependencies/requirements.txt
mkdir data
mkdir logs
mkdir model_dir
