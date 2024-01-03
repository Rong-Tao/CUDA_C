#include <iostream>
#include <stdio.h>
#include <thrust/set_operations.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

extern "C" {
    int launchFindIntersection(int* list1, int* list2, int list1Size, int list2Size, int* result) {
        thrust::device_vector<int> d_list1(list1, list1 + list1Size);
        thrust::device_vector<int> d_list2(list2, list2 + list2Size);
        thrust::device_vector<int> d_result(list1Size + list2Size);

        auto d_result_end = thrust::set_union(thrust::device, 
                                            d_list1.begin(), d_list1.end() , 
                                            d_list2.begin(), d_list2.end() , 
                                            d_result.begin(), 
                                            thrust::greater<int>());
        thrust::copy(d_result.begin(), d_result_end, result);
        return d_result_end - d_result.begin(); 
    }
} 
