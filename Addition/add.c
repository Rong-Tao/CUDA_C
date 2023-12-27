// addition.c
#include <stdio.h>

int add(int a, int b) {
    printf("C => Add: %d and %d\n", a, b);  // Debug print
    int result = a + b;
    printf("C => Result: %d\n", result);  // Debug print
    fflush(stdout);
    return result;
}
