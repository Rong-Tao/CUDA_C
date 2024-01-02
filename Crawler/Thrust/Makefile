# Compiler and Flags
NVCC = nvcc
NVCCFLAGS = -Xcompiler -fPIC
LDFLAGS = -shared

# Target
TARGET = findIntersection

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

# New target 'run' for cleaning and recompiling
run: clean all

.PHONY: all clean run
