CXX      = g++
CXXFLAGS = -O2 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter

TARGET = solve
SRCS   = main.cpp solve.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) solve.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
