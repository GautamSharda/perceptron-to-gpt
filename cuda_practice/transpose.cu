// pmpp ch4 problems 2 and 3
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
