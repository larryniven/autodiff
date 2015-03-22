CXXFLAGS += -std=c++11 -I ../../

.PHONY: all clean

all: test_autodiff

clean:
	-rm *.o
	-rm test_autodiff

test_autodiff: test_autodiff.o ../autodiff.o
	$(CXX) $(CXXFLAGS) -o $@ $^

