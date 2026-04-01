CC := gcc
CFLAGS := -O2 -std=c11 -Wall -Wextra -pedantic

all: stack_vm

stack_vm: main.o
	$(CC) $(CFLAGS) -o $@ main.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f main.o stack_vm

.PHONY: all clean
