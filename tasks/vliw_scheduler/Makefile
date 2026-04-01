CC      = gcc
CFLAGS  = -O2 -march=native -Wall -Wextra -std=c11
TARGET  = solve

all: $(TARGET)

$(TARGET): main.c solve.c solve.h
	$(CC) $(CFLAGS) -o $@ main.c solve.c

clean:
	rm -f $(TARGET)
