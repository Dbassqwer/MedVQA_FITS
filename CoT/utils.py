import torch
from string import Template
from collections import namedtuple
import cupy

Stream = namedtuple('Stream', ['ptr'])

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
    # elif isinstance(t, torch.cuda.HalfTensor):
    #     return 'float'
@cupy.memoize(for_each_device=False)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    # print('kernel_name:',kernel_name)
    # print('code:',code)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

