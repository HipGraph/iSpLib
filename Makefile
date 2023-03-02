all:
	sh ./configure
	pip install -e .

test:
	python3 ./tests/GCN.py
