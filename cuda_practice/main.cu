// Multiblock MatMul with flattened Matrices.
#include <stdio.h>
#include <cuda_runtime.h>

// Let's first practice doing it on CPU
__host__ void matmul_kernel_cpu(float* Md, float* Nd, float* Pd, int Width){
    // Assume same dims and no threads for a moment
    // Width is therefore one side (row or col) of the matrices, all of same width
    // But assume flattened matrices
    for (int i = 0; i < Width; i++){
        // iterate through the rows vectors of the first matrix
        for (int j = 0; j < Width; j++){
            // iterate through columns of the second matrix
            float dp = 0.0f;
            for (int k = 0; k < Width; k++){
                // iterate through the elements of each vector in both matrices
                float r = Md[i*Width + k]; // row-wise for the first matrix
                float c = Nd[k*Width + j]; // column-wise for the second matrix
                dp += r*c; // summate the products
            }
            // place dot product in result matrix
            Pd[i*Width + j] = dp; // at the ith row and jth col.
        }
    }

    // Print the resulting matrix Pd from within the kernel.
    printf("\nResult Matrix Pd (from kernel):\n");
    for(int i = 0; i < Width; i++) {
        for(int j = 0; j < Width; j++) {
            printf("%.1f ", Pd[i * Width + j]);
        }
        printf("\n");
    }
}

__global__ void matmul_kernel_gpu_singleblock(float* Md, float* Nd, float* Pd, int Width){
    // This will be one of the threads in the thread block -- specifically at the following
    // These correspond to the row / column in the Md and Nd matrices
    int i = threadIdx.x; // This is the col in Pd (and therefore also in Nd)
    int j = threadIdx.y; // This is the row in Pd (and therefore also in Md)
    float dp = 0.0f;
    for (int k = 0; k < Width; k++){ // k remains the column in Md and row in Nd
        // Applying index = row*width + col
        float r = Md[j*Width + k]; // row-wise for the first matrix
        float c = Nd[k*Width + i]; // column-wise for the second matrix
        dp += r*c; // summate the products
    }
    Pd[j*Width + i] = dp; // You can't just append I guess? so you apply row*width + col
}

__global__ void matmul_kernel_gpu_multiblock(float* Md, float* Nd, float* Pd, int Width){
    // Don't assume any specific Width, but assume it's greater than BlockDim.x == BlockDim.y --> Multiple blocks == (GridDim =/= 1x1)
    // --> Tiling --> Bounds checking! 
    int row = blockIdx.y*blockDim.y + threadIdx.y; // Global Thread ID, which can be used for indexing. Specifically, this is the row in Pd (and therefore also in Md).
    int col = blockIdx.x*blockDim.x + threadIdx.x; // Global Thread ID, which can be used for indexing. Specifically, this is the col in Pd (and therefore also in Nd).
    if (row < Width && col < Width){ // Skip if out of bounds
        float dp = 0.0f;
        for (int k = 0; k < Width; k++){ // For indexing within the vectors. This is a col in Md and a row in Nd. 
            // indexing in flattened matrix: flat_index = row_of_element_in_OG_Matrix*width_of_Matrix + col_of_element_in_OG_matrix (!) i.e., index = row*width + col
            dp += Md[row*Width + k] * Nd[k*Width + col];
        }
        // Again, same indexing strategy. If you name every variable appropriately and understand what it represents, this becomes clear.
        Pd[row*Width + col] = dp;
    }
}

void matmul(float* M, float* N, float* P, int Width){
    // Create pointers for new memory on device and allocate the new memory 
    float* Md;
    int sizeM = Width*Width*sizeof(float);
    cudaMalloc((void**)&Md, sizeM);
    float* Nd;
    int sizeN = Width*Width*sizeof(float);
    cudaMalloc((void**)&Nd, sizeN);
    float* Pd;
    int sizeP = Width*Width*sizeof(float);
    cudaMalloc((void**)&Pd, sizeP);
    printf("Memory allocated.\n");
    // Copy the data to the new memory addresses given to the pointers by cudaMalloc()
    cudaMemcpy(Md, M, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, sizeN, cudaMemcpyHostToDevice);
    // Do the MatMul
    // matmul_kernel_cpu(M, N, P, Width); // CPU version
    dim3 dimBlock(2, 2);
    dim3 dimGrid(1, 1);
    matmul_kernel_gpu_multiblock<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width); // GPU version
    // Free the previously allocated device memory
    cudaMemcpy(P, Pd, sizeP, cudaMemcpyDeviceToHost);
    // Print the matrix P from the host right after copying it back from the device.
    printf("\nResult Matrix P (from matmul, after GPU computation):\n");
    for(int i = 0; i < Width; i++) {
        for(int j = 0; j < Width; j++) {
            printf("%.1f ", P[i * Width + j]);
        }
        printf("\n");
    }

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
    printf("Memory freed.\n");
}

int main(){
    float M[4] = {1, 2, 3, 4};
    float N[4] = {5, 6, 7, 8};
    float P[4];
    matmul(M, N, P, 2); // M is converted to &M[0]
}