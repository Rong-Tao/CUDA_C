#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel for vector addition
__global__ void sumKernel(float *a, float *b, float *c, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        int index = row * width + col;
        c[index] = a[index] + b[index];
    }
}


// Host function to launch the kernel
extern "C" void kernel_launcher(float *a, float *b, float *c, int height, int width) {
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, 
                   (height + blockSize.y - 1) / blockSize.y);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, width * height * sizeof(float));
    cudaMalloc(&d_b, width * height * sizeof(float));
    cudaMalloc(&d_c, width * height * sizeof(float));

    cudaMemcpy(d_a, a, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, width * height * sizeof(float), cudaMemcpyHostToDevice);

    sumKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, height, width);

    cudaMemcpy(c, d_c, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
