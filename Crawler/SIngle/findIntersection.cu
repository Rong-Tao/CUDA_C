#include <iostream>
#include <cuda_runtime.h>
using namespace std;


__global__ void findIntersection(int *list1, int *list2, int list1Size, int list2Size, int deltax, int deltay, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < list1Size) {
        int x = list1[idx * 2] + deltax;
        int y = list1[idx * 2 + 1] + deltay;
        
        for (int i = 0; i < list2Size; ++i) {
            if (x == list2[i * 2] && y == list2[i * 2 + 1]){
                result[idx] = 1;
                return;
            }
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


    int blockSize = 256;
    int numBlocks = (list1Size + blockSize - 1) / blockSize; // Ensure there are enough blocks to cover all elements

    // Launch the kernel
    findIntersection<<<numBlocks, blockSize>>>(d_list1, d_list2, list1Size, list2Size, deltax, deltay, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_list1);
    cudaFree(d_list2);
    cudaFree(d_result);
}