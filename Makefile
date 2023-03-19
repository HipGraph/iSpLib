all:
	git submodule update --init --recursive
	pip install -e .

test:
	python3 ./tests/GCN.py
