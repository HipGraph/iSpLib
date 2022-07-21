python -m venv venv
.\venv\Scripts\activate
pip install torch
download from link https://github.com/rusty1s/torch_sparse/archive/refs/tags/0.6.14.zip
cd the_sparse_package
python setup.py develop
Replace the name with the new name
cd ..
cd pytorch_geometric
python setup.py develop