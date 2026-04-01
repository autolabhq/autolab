# Makefile for radix_sort
# DO NOT MODIFY THIS FILE.

CC     = gcc
CFLAGS = -O2 -std=c99 -Wall -Wextra -pedantic
TARGET = solve
SRCS   = main.c solve.c
HDRS   = solve.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
