TEMP_PATH = ./kernels/tmp

all:
	nvcc -Xcompiler -fPIC  -rdc=true -c -o $(TEMP_PATH)/temp.o fusedmm.cu
	nvcc -Xcompiler -fPIC -dlink -o $(TEMP_PATH)/temp2.o $(TEMP_PATH)/temp.o -lcudart
	ar cru ./bin/fusedmm_gpu.a $(TEMP_PATH)/*.o
	ranlib ./bin/fusedmm_gpu.a
	g++ timer.cpp -L./bin -l:fusedmm_gpu.a -o timer -L/usr/local/cuda/lib64 -lcudart
	@echo Static Library created in ./bin. Copy this to iSpLib/csrc/fusedmm and run python3 setup.py develop there.
codegen:
	@echo Auto-tuning...
	python3 ./kernels/codegen.py
	@echo Done!
test:
	./timer
	
clean:
	rm -f ./bin/* ./kernels/tmp/* ./kernels/generated/include/spmm_header.cuh ./timer
	touch ./kernels/generated/include/spmm_header.cuh
	
