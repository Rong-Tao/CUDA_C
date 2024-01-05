#include <iostream>
#include <cuda_runtime.h>
using namespace std;


__global__ void findIntersection(int *list1, int *list2, int list1Size, int list2Size, int deltax, int deltay, int *result) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < list1Size && idx_y < list2Size) {
        int x = list1[idx_x * 2] + deltax;
        int y = list1[idx_x * 2 + 1] + deltay;

        if (x == list2[idx_y * 2] && y == list2[idx_y * 2 + 1]) {
            result[idx_x] += 1; // Mark as found
        }
    }
}



extern "C" void launchFindIntersection(int *h_list1, int *h_list2, int list1Size, int list2Size, int deltax, int deltay, int *h_result) {
    int *d_list1, *d_list2, *d_result;
    int size1 = list1Size * 2 * sizeof(int);
    int size2 = list2Size * 2 * sizeof(int);
    int resultSize = list1Size * sizeof(int);

    cudaMalloc((void**)&d_list1, size1);
    cudaMalloc((void**)&d_list2, size2);
    cudaMalloc((void**)&d_result, resultSize);

    cudaMemcpy(d_list1, h_list1, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_list2, h_list2, size2, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, resultSize); 


    dim3 blockSize(16, 16); // Example block size, adjust as needed
    dim3 numBlocks((list1Size + blockSize.x - 1) / blockSize.x,
                   (list2Size + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    findIntersection<<<numBlocks, blockSize>>>(d_list1, d_list2, list1Size, list2Size, deltax, deltay, d_result);
    cudaDeviceSynchronize();


    cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_list1);
    cudaFree(d_list2);
    cudaFree(d_result);
}