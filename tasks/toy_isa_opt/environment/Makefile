# Makefile for toy_isa_opt
# DO NOT MODIFY THIS FILE.

CC     = gcc
CFLAGS = -O2 -std=c99 -Wall -Wextra
TARGET = solve

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
