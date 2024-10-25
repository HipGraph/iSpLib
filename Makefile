all:
	pip install .

test:
	python3 ./tests/GCN.py

clean:
	rm -frd isplib/*.so build/*
