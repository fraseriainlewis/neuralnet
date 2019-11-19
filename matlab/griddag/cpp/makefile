# use "gcc" to compile source files.
CC = clang++
# the linker is also "gcc". It might be something else with other compilers.
LD = clang++
# Compiler flags go here.
#CFLAGS = -I/home/fraser/R/lib64/R/include -I/home/fraser/include -Wall -pedantic -O3 -fPIC 

CFLAGS = -std=c++17 -Wall -pedantic -O3 

# Linker flags go here. Currently there aren't any, but if we'll switch to
#LDFLAGS = -shared  -L/home/fraser/lib -lgsl -lgslcblas 

LDFLAGS = 

# use this command to erase files.
RM = /bin/rm -f
# list of generated object files.
OBJS =  envDAG.o main.o 

# program executable file name.
PROG = a.out

# top-level rule, to compile everything.
all: $(PROG)

# rule to link the program - NEW NOTE: $(LDFLAGS) must come last with gcc 4.4 onwards
$(PROG): $(OBJS)
	$(LD) $(OBJS) -o $(PROG) $(LDFLAGS)

envDAG.o: envDAG.cpp envDAG.hpp
	$(CC) $(CFLAGS) -c envDAG.cpp

main.o: main.cpp 
	$(CC) $(CFLAGS) -c main.cpp


# rule for cleaning re-compilable files.
clean:
	$(RM) $(PROG) $(OBJS)

