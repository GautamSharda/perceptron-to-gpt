// pmpp ch4 problem 1: I would tell the student who claimed to have done a tiled matmul of 2 1024x1024 matrices with 1024 blocks on a G80 that he is confused or lying because the G80 supports only 512 threads / block and the stated configuration would require 1024 threads / block, so the kernel should fail to launch. Though if I am limited only to the information mentioned in chapter 4, it would be somewhat tricky to infer this limit: The only way I see to infer it is the chapter mentions the G200 only supports 512 threads / block, so I guess I am to assume older gens don't support more.

// pmpp ch4 problems 2 and 3
// First of all I thought this was interesting because the tranpose is fancier than it initially seemed because of the way it uses shared memory to save global memory accesses. At first I thought it just wrote to global memory (yes, stupid, because it says "shared" but I assumed that meant shared by all blocks -- due to my lack of hardware intuition a la SMs sharing execution resources and warp switching for latency hiding, so I do see now why it's a silly assumption -- and also plainly stupid because at that point why not just write directly to the original matrix). However, instead it does a local transpose, and then writes the local tile to the original matrix in it's *transpose* location (which is at least a mildly fancy shape rotation algorithm so maybe you can forgive me for visualizaing it incorrectly initially).
// Anyway, as for the reason this code fails without the sync is pretty simple once you understand the basic fact that not all threads execute in the same clock cycle but rather they are grouped into warps (usually packs of 32. and it's also worth noting my homeboy wecu told me that even "lock step" within warps is kind of false due to divergent branches being scheduled separately). Further, warp scheduling is not determinsitic so you could have a case where if you have multiple warps in a block then the read for some of your local transpose could happen before or at the same time as the corresponding writes. Of course, this isn't the issue with only 1 warp, so BLOCK_SIZE < 6 may not necessarily require it. 6*6 = 36 and above block size would though so the answer is <6.    
#include <stdio.h>
#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void BlockTranspose(float* A_elements, int A_width, int A_height) {
    __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE]; 
    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;
    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    __syncthreads(); // crucial to prevent race condition
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

template <int BLOCK_SIZE>
void transpose(float* A, int A_height, int A_width){
    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y); 
    BlockTranspose<BLOCK_SIZE><<<gridDim, blockDim>>>(A, A_width, A_height);
}

int main(){
    float A[16] = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
       13.0f, 14.0f, 15.0f, 16.0f
    };
    transpose<2>(A, 4, 4);
}
