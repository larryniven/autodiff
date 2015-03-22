CXXFLAGS += -std=c++11 -I ../

.PHONY: all clean

all: autodiff.o

clean:
	-rm *.o

autodiff.o: autodiff.h
