CFLAGS=-g -O2 -Wall --std=c99
TARGETS=gaus_pthread

all: $(TARGETS)

gaus_pthread: gaus_pthread.c
	gcc $(CFLAGS) -o $@ $< -lpthread -lm

clean:
	rm -f $(TARGETS)
