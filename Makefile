all:
	pip install -e .

test:
	python3 ./tests/GCN.py

clean:
	rm -frd isplib/*.so build/*