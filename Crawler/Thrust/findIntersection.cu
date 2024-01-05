#include <thrust/transform.h>
#include <thrust/set_operations.h>
#include <thrust/device_vector.h>
#include <iostream>

void printvec(const thrust::device_vector<int>& list){
    for (const auto& element : list) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

void transformIntersectAndCopy(const thrust::device_vector<int>& d_list1, 
                               const thrust::device_vector<int>& d_list2, 
                               int direction, // Use an integer to specify the direction
                               int width, 
                               int* result, 
                               int& resultSize) {
    thrust::device_vector<int> transformed(d_list1.size());
    thrust::device_vector<int> d_result(d_list1.size());

    thrust::transform(d_list1.begin(), d_list1.end(), transformed.begin(), 
                      [width, direction] __device__ (int x) { 
                          switch (direction) {
                              case 0: return x + 2;           // Up
                              case 1: return x - 2;           // Down
                              case 2: return x + 2 * width;   // Left
                              case 3: return x - 2 * width;   // Right
                              default: return x;
                          }
                      });
    //std::cout << "L1: " << std::endl;
    //printvec(transformed);
    
    
    auto end = thrust::set_intersection(thrust::device, transformed.begin(), transformed.end(), 
                                        d_list2.begin(), d_list2.end(), d_result.begin()); 

    thrust::copy(d_result.begin(), d_result.end(), result);
    //printvec(d_result);
    //std::cout << "------------------" << std::endl;

    resultSize =  end - d_result.begin();
}
extern "C" {
    void launchFindIntersection(int* list1, int* list2, int list1Size, int list2Size, int width, 
                                int* resultSizes, int* upResult, int* downResult, int* leftResult, int* rightResult) {
        thrust::device_vector<int> d_list1(list1, list1 + list1Size);
        thrust::device_vector<int> d_list2(list2, list2 + list2Size);

        thrust::sort(d_list1.begin(), d_list1.end());
        thrust::sort(d_list2.begin(), d_list2.end());

        transformIntersectAndCopy(d_list1, d_list2, 0, width, upResult, resultSizes[0]); // Up
        transformIntersectAndCopy(d_list1, d_list2, 1, width, downResult, resultSizes[1]); // Down
        transformIntersectAndCopy(d_list1, d_list2, 2, width, leftResult, resultSizes[2]); // Left
        transformIntersectAndCopy(d_list1, d_list2, 3, width, rightResult, resultSizes[3]); // Right
    }
}
