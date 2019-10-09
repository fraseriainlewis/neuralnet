# use "gcc/clang" to compile source files.
# note below flags were found by using R CMD SHLIB utility.c and R CMD SHLIB utility -lgsl -lgslclbas for the correct format for clang
CC = clang
# the linker is also "gcc". It might be something else with other compilers.
LD = clang 
# Compiler flags go here.
CFLAGS = -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG -I/usr/local/include -fPIC  -Wall -g -O2
LDFLAGS1 = -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -single_module -multiply_defined suppress -L/Library/Frameworks/R.framework/Resources/lib -L/usr/local/lib 
LDFLAGS2 = -lgsl -lgslcblas -F/Library/Frameworks/R.framework/.. -framework R -Wl,-framework -Wl,CoreFoundation  
# use this command to erase files.
RM = /bin/rm -f
# list of generated object files.
OBJS =  utility.o cycles.o

# program executable file name.
PROG = abn.so

# top-level rule, to compile everything.
all: $(PROG)

# rule to link the program - NEW NOTE: $(LDFLAGS) must come last with gcc 4.4 onwards
$(PROG): $(OBJS)
	$(LD) $(LDFLAGS1) -o $(PROG) $(OBJS) $(LDFLAGS2)

utility.o: utility.c structs.h
	$(CC) $(CFLAGS) -c utility.c

cycles.o: cycles.c structs.h
	$(CC) $(CFLAGS) -c cycles.c

# rule for cleaning re-compilable files.
clean:
	$(RM) $(PROG) $(OBJS)

