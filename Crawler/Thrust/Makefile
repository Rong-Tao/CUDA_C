# Compiler and Flags
NVCC = nvcc
NVCCFLAGS = -Xcompiler -fPIC --expt-extended-lambda
LDFLAGS = -shared

# Target
TARGET = findIntersection.so

# Source files
SOURCE = findIntersection.cu

# Default target
all: $(TARGET)

# Compile the CUDA code into a shared library
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $(TARGET) $(SOURCE)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean
