CC       = cc
CFLAGS   = -O2 -march=native -Wall -Wextra -std=c99
LDFLAGS  = -lm

.PHONY: all clean

all: evolve

evolve: evolve.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f evolve
