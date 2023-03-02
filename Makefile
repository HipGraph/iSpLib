all:
	sh ./configure
	pip install -e .

test:
	python3 ./test/GCN.py