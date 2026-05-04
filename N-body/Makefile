CC       = cc
CFLAGS   = -O2 -march=native -fPIC -Wall -Wextra -std=c99
LDFLAGS  = -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB = nbody_core.dylib
    SOFLAGS = -dynamiclib
else
    LIB = nbody_core.so
    SOFLAGS = -shared
endif

.PHONY: all clean omp

all: $(LIB)

$(LIB): nbody_core.c nbody_core.h
	$(CC) $(CFLAGS) $(SOFLAGS) -o $@ nbody_core.c $(LDFLAGS)

# Build with OpenMP (non-deterministic BH acceleration due to reduction
# order, but faster for production runs).
omp: CFLAGS += -fopenmp -D_OPENMP
omp: LDFLAGS += -fopenmp
omp: $(LIB)

clean:
	rm -f nbody_core.so nbody_core.dylib
