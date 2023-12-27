# CUDA Playground

This repository contains files that primarily focus on running Python code in conjunction with C and CUDA (C-based) code.

## Addition

In the Addition section, we explore a simple numerical addition operation. It's noteworthy that for compiling the C code, we use `nvcc` (NVIDIA CUDA Compiler)

The asynchronous behavior of print statements between Python and C, especially when a C function is called from Python, arises due to differences in how each language handles output buffering:

- **Python's Print Function**: Automatically flushes the output buffer each time it's called, typically displaying the output immediately.
- **C's printf Function**: May not immediately flush the output buffer, particularly when output is redirected or part of a shared library. This can lead to delayed or batched display of output.
When Python calls a C function, these differences can result in out-of-order or seemingly asynchronous output between the two, as the timing of when outputs are actually written to the terminal or console can vary.