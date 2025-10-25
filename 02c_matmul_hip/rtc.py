import torch
import time


kernel_cu = r"""
__global__
void add_kernel(const int *A, const int *B, int *C, int M, int N) {
  // A, B, C are 2D matrices with shape M, N
  // we will use 1 threadblock to handle 1 row
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tb_size = blockDim.x;

  A += bid * N;
  B += bid * N;
  C += bid * N;

//printf("block_idx: %d thread_idx: %d   A:%p B:%p C:%p here...\n", blockIdx.x, threadIdx.x, A, B, C);
  for (int col = tid; col < N; col += tb_size)
    C[col] = A[col] + B[col];
}
"""

tic = time.perf_counter()
ret_kernel = torch.cuda._compile_kernel(kernel_cu, "add_kernel")
toc = time.perf_counter()
print(f"compile used {toc - tic}")


M, N = 100, 1024
a = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)
b = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)

def add2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    C = torch.empty_like(A)
    print("start_kernel here", A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N)
    ret_kernel((1, 1, 1), (30, 1, 1), (A.data_ptr(), B, C, M, N))
    print("end_kernel here")
    torch.cuda.synchronize()
    print("synchronize done")
    return C
    


torch.testing.assert_close(add2(a, b ), a + b)
