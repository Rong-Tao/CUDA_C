# CUDA Playground

This repository contains files that primarily focus on running Python code in conjunction with C and CUDA (C-based) code.

## Addition

In the Addition section, we explore a simple numerical addition operation. It's noteworthy that for compiling the C code, we use `nvcc` (NVIDIA CUDA Compiler)

The asynchronous behavior of print statements between Python and C, especially when a C function is called from Python, arises due to differences in how each language handles output buffering:

- **Python's Print Function**: Automatically flushes the output buffer each time it's called, typically displaying the output immediately.
- **C's printf Function**: May not immediately flush the output buffer, particularly when output is redirected or part of a shared library. This can lead to delayed or batched display of output.
When Python calls a C function, these differences can result in out-of-order or seemingly asynchronous output between the two, as the timing of when outputs are actually written to the terminal or console can vary.

## Addition_CUDA

In this example, we demonstrate the integration of CUDA with Python for performing an addition operation. The code consists of three main components:

1. **CUDA Kernel (`sumKernel`):**
   - This kernel is written in CUDA C and is responsible for performing the addition operation. Each thread in the kernel adds its index to a shared result variable using `atomicAdd` to ensure thread safety. A print statement is included in the kernel for the first thread to output the intermediate sum.

2. **C Function (`kernel_launcher`):**
   - Defined with `extern "C"` to prevent name mangling, this function acts as a bridge between the CUDA kernel and the Python environment. It handles memory allocation on the GPU, launches the CUDA kernel, and then copies the result back to the host memory. After executing the kernel, it prints the final sum from the C perspective.

3. **Python Integration:**
   - Using Python's `ctypes` library, we load the compiled CUDA code as a shared library. The `kernel_launcher` function is set up with appropriate argument and return types, and is then called from Python. This showcases how Python can leverage CUDA for high-performance computations.

### Execution Flow:

1. The Python script calls `kernel_launcher` with the number of threads.
2. `kernel_launcher` allocates memory, launches the `sumKernel`, and retrieves the result.
3. The `sumKernel` performs the addition in parallel on the GPU.
4. Outputs from both the CUDA kernel and the C function are printed, followed by the final result printed in Python.

This example highlights the synergy between Python and CUDA, enabling efficient computations with parallel processing capabilities of GPUs.
