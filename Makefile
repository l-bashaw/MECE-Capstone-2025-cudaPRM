INC_DIR = ./include
SRC_DIR = ./src
OBJ_DIR = ./obj
CC = nvcc
CFLAGS = -I$(INC_DIR)
LDFLAGS = -lcublas

# Define source and object files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_FILES))
INC_FILES = $(wildcard $(INC_DIR)/*.h)

# Define main executable
MAIN_EXE = main

DEPS = $(INC_FILES)

# Default rule (build main executable)
all: $(OBJ_DIR) $(MAIN_EXE)

# Compile main executable
$(MAIN_EXE): $(OBJ_FILES) $(DEPS)
	$(CC) $(CFLAGS) -o $@ $(OBJ_FILES) $(LDFLAGS)

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	$(CC) -c $< -o $@ $(CFLAGS)

# Clean up
clean:
	rm -f $(MAIN_EXE) $(OBJ_DIR)/*.o