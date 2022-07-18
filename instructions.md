python -m venv venv
.\venv\Scripts\activate
pip install torch
cd the_sparse_package
python setup.py develop
cd ..
cd pytorch_geometric
python setup.py develop