import lib.deviceQuery as device
import os
curPath = os.path.dirname(__file__)

# curPath = Path(__file__).parent.absolute()

# print(device.getDeviceInfo())





# blc_per_row = 256

with open(curPath + '/templates/spmm.template') as f:
    code_template = f.read()

with open(curPath + '/templates/spmm.param') as f:
    param = eval(f.read())

if param['end'] == '[[MAX_THREAD]]':
    param['end'] = int(device.getDeviceInfo()[0]['MAX_THREADS_PER_BLOCK'])
    # print(param['end'])

# print(param['start'])
functions = []
prototypes = []
function_names = []

for i in range(int(param['start']), int(param['end']) + 1, int(param['step'])):
    # print(code_template.replace(param['name'], str(i)))
    functions += [code_template.replace(param['name'], str(i))]
    prototypes += [param['prototype'].replace(param['name'], str(i))]
    function_names += [param['function_names'].replace(param['name'], str(i))]

code_file_content = '\n\n'.join(functions)
header_file_content = param['typedef'] + '\n\n'.join(prototypes) + '\n\n' + param['array_pointer'].replace('[[function_names]]', ',\n'.join(function_names))
all_file_content = param['typedef'] + '\n\n'.join(prototypes) + '\n\n'.join(functions) + '\n\n' + param['array_pointer'].replace('[[function_names]]', ',\n'.join(function_names))

with open(curPath + '/generated/include/spmm_header.cuh', 'w') as f:
    f.write(all_file_content)

# print(param['typedef'])
# print('\n'.join(prototypes))
# print('\n\n'.join(functions))
# print(param['array_pointer'].replace('[[function_names]]', ',\n'.join(function_names)))
# print(file_content)


# header_template = r'''void fusedmm_cuda_bl{{blc_per_row}}(int m, int n, int k, int nnz, const int64_t* indx, const int64_t* ptrb, const float* val, const float* b, float* c);'''

# headers = []
# for k in range(128, 513, 8):
#     with open(f'generated/spmm_kernel_bl{k}.cu', 'w') as f:
#         f.write(code_template.replace('{{blc_per_row}}', k))
#         headers += [header_template.replace('{{blc_per_row}}', k)]
    
#     with open('kernel.cuh', 'w') as f:
#         f.write('\n'.join(headers))