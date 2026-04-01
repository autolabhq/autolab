# Makefile for sha256_throughput
# DO NOT MODIFY THIS FILE.

CC     = gcc
CFLAGS = -O2 -march=native -std=c99 -Wall -Wextra
TARGET = solve
SRCS   = main.c solve.c
HDRS   = solve.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
