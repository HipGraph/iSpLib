all:
	git submodule update --init --recursive
	pip3 install -e .

test:
	python3 ./tests/GCN.py
