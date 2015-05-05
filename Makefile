CXXFLAGS += -std=c++11 -I ../
AR = gcc-ar

.PHONY: all clean

all: libautodiff.a

clean:
	-rm *.o
	-rm libautodiff.a

libautodiff.a: autodiff.o
	$(AR) rcs $@ $^

autodiff.o: autodiff.h
