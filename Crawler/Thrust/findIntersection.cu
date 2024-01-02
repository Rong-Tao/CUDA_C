#include <iostream> // Required for std::cout
#include <stdio.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

int main() { // Corrected syntax here

    int A1[7] = {12, 10, 8, 6, 4, 2, 0};
    int A2[5] = {9, 7, 5, 3, 1};
    int result[12]; // Adjusted size to 12, since 7 + 5 = 12

    // thrust::set_union returns an iterator to the end of the result range
    int *result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());

    // Calculate the size of the result
    int result_size = result_end - result;

    // Print the result
    std::cout << "Union result: [";
    for (int i = 0; i < result_size; ++i) {
        std::cout << result[i];
        if (i < result_size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    return 0;
}
