- Update the `fusedmm_spmm_trusted_kernel` in `./kernels/spmm.cuh`. Please keep the function prototype unchanged.
- Run `make all`. Then copy the static library generated in ./bin folder to iSpLib/csrc/fusedmm/ and run `python3 setup.py develop` in iSpLib.
- To test timing, run `make test`

