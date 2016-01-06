CXXFLAGS += -std=c++11 -I ../../ -L ../../ebt -L ../../la

.PHONY: all gpu clean

all: test-autodiff

gpu: test-autodiff-gpu test-autodiff-op-gpu

clean:
	-rm *.o
	-rm test_autodiff

test-autodiff: test-autodiff.o ../autodiff.o ../autodiff-op.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lla -lebt -lcblas

test-autodiff-gpu: test-autodiff-gpu.o ../autodiff.o ../autodiff-gpu.o ../autodiff-op.o ../autodiff-op-gpu.o
	nvcc $(CXXFLAGS) -o $@ $^ -llagpu -lebt -lcublas -lcblas

test-autodiff-op-gpu:: test-autodiff-op-gpu.o ../autodiff-op-gpu.o
	nvcc $(CXXFLAGS) -o $@ $^ -llagpu -lebt -lcublas

test-autodiff-gpu.o: test-autodiff-gpu.cu
	nvcc $(CXXFLAGS) -c test-autodiff-gpu.cu

test-autodiff-op-gpu.o: test-autodiff-op-gpu.cu
	nvcc $(CXXFLAGS) -c test-autodiff-op-gpu.cu
