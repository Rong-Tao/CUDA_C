#include <stdio.h>

__global__ void sumKernel(int *result) {
    int index = threadIdx.x + 1; 
    atomicAdd(result, index);
    if (index==1){
        printf("Kernel => Sum = %d\n", *result);
    }
}
extern "C" {
int kernel_launcher(int i){
    int *result;
    int *d_result;
    int sum = 0;

    result = &sum;

    cudaMalloc((void **)&d_result, sizeof(int));
    cudaMemcpy(d_result, result, sizeof(int), cudaMemcpyHostToDevice);
    sumKernel<<<1, i>>>(d_result);
    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    printf("C => Sum = %d\n", *result);

    return *result;
}
}