CC      = gcc
CFLAGS  = -O2 -std=c11 -march=native -Wall -Wextra
LDFLAGS = -lm

TARGET  = attn

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.c solve.c solve.h
	$(CC) $(CFLAGS) -o $@ main.c solve.c $(LDFLAGS)

clean:
	rm -f $(TARGET)
