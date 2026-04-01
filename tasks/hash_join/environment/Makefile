# Makefile — DO NOT MODIFY

CC      = gcc
CFLAGS  = -O2 -std=c99 -Wall -Wextra -march=native
LDFLAGS =

TARGET  = join
SRCS    = main.c solve.c

$(TARGET): $(SRCS) solve.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
