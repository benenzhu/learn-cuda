import torch
from typing import Optional, Union, Any
import ctypes
import sys
from rtc import _compile_kernel
from triton.testing import do_bench
from contextlib import nullcontext

import time
print(torch.__version__)
with open("kernel.cu", "r") as f:
    KERNEL_SOURCE = f.read()
tic = time.time()
kernel = _compile_kernel(
    KERNEL_SOURCE,
    kernel_name="add_kernel",
)
toc = time.time()

print("compile used time: ", toc - tic)

def div_up(x):
    return (x + 255) // 256

def get_torch_prof_ctx(do_perf = False):
    ctx = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False if torch.cuda.device_count() == 1 else False,
    ) 
    return ctx if do_perf else nullcontext()



def bench():
    ctx =  get_torch_prof_ctx()
    with ctx:
        for exp in range(28, 30):
            M = 2 ** exp

            input1 = torch.randn(M, device="cuda")
            input2 = torch.randn(M, device="cuda")
            output = torch.empty(M, device="cuda")
            def call():
                return kernel((div_up(M//4),1,1), (256,1,1), (input1, input2, output, input1.numel()))
            call()

            torch.testing.assert_close(output, input1 + input2)
            
            tic = do_bench(call, warmup=100, rep=500)
            print("tic", tic)
            bandwidth_GB = M * 4 * 2 / (tic * 1e-3) / 1e3 / 1e3 /1e3 # KB -> MB -> GB
            print(f"M={M:10,}, bandwidth_GB={bandwidth_GB:10.2f} GB/s, ms: {tic:10.2f} ms, block_num: {div_up(M//4):10,}")

    if type(ctx) == torch.profiler.profile:
        ctx.export_chrome_trace(f"00.json")
        


def get_kernel(kernel_name, file_name="kernel.cu"):
    tic = time.time()
    kernel = _compile_kernel(
        open(file_name, "r").read(),
        kernel_name=kernel_name,
        nvcc_options=["-std=c++17", "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"],
    )
    toc = time.time()
    print("compile used time: ", toc - tic)
    return kernel


def test_async_cp_kernel():
    async_cp_kernel = get_kernel("async_cp_kernel")
    M = 16 * 16
    a = torch.arange(M, device="cuda").half()
    async_cp_kernel((1,1,1), (32,1,1), (a,))
    torch.cuda.synchronize()
    time.sleep(0.5)
 
# test_async_cp_kernel()

def test_ld_matrix_kernel():
    ld_matrix_kernel = get_kernel("ld_matrix_kernel")
    a = torch.zeros(16, 32, device="cuda").half()
    ld_matrix_kernel((1,1,1), (32,1,1), (a,))
    torch.cuda.synchronize()
    time.sleep(0.5)
    
# test_ld_matrix_kernel()

def test_mma_ptx_kernel():
    mma_ptx_kernel = get_kernel("mma_ptx_kernel", file_name="02_mma_ptx.cu")
    a = torch.arange(16 * 16, device="cuda").half() * 0.1
    b = torch.arange(16 * 16, device="cuda").half() * 0.1
    c = torch.zeros(16 * 16 * 16, device="cuda").half()
    d = torch.matmul(a.reshape(16, 16), b.reshape(16, 16).T)
    mma_ptx_kernel((1,1,1), (32,1,1), (c, a, b, d))
    torch.cuda.synchronize()
    time.sleep(0.5)

# test_mma_ptx_kernel()

def test_tma_1d_kernel():
    tma_1d_kernel = get_kernel("tma_1d_kernel", file_name="02_mma_ptx.cu")
    warps_per_block = 4 
    threads_per_warps = 32
    elts_per_threads = 8
    elts = warps_per_block * threads_per_warps * elts_per_threads
    a = torch.arange(elts, device="cuda").half() * 0.1
    tma_1d_kernel((1,1,1), (warps_per_block * threads_per_warps, 1, 1), (a, a.numel()), shared_mem=2 * elts)
    torch.cuda.synchronize()
    time.sleep(0.5)

test_tma_1d_kernel()

def get_mma_kernel():
    # with open("kernel.cu", "r") as f:
    #     KERNEL_SOURCE = f.read()
    tic = time.time()
    kernel = _compile_kernel(
        open("kernel_mma.cu", "r").read(),
        kernel_name="mma_kernel",
    )
    toc = time.time()
    print("compile used time: ", toc - tic)
    return kernel


