# Makefile for levenshtein_distance
# DO NOT MODIFY THIS FILE.

CC     = gcc
CFLAGS = -O2 -march=native -std=c11 -Wall -Wextra
TARGET = solve
SRCS   = main.c solve.c
HDRS   = solve.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
