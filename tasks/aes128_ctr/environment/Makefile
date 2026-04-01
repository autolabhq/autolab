CC      = gcc
CFLAGS  = -O2 -march=native -std=c99 -Wall -Wextra
TARGET  = solve

all: $(TARGET)

$(TARGET): main.c solve.c solve.h
	$(CC) $(CFLAGS) -o $@ main.c solve.c

clean:
	rm -f $(TARGET)
