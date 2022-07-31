create a folder for the project


git clone https://github.com/pyg-team/pytorch_geometric.git 


download from link https://github.com/rusty1s/pytorch_sparse/archive/refs/tags/0.6.14.zip


replace the word torch_sparse with the_sparse_package


replace the word pythe_sparse_package with pytorch_sparse


python -m venv venv


.\venv\Scripts\activate

For mac: source venv/bin/activate


pip install torch


cd the_sparse_package


rm .git*


pip install .


cd ..


cd pytorch_geometric


rm .git*


pip install .
