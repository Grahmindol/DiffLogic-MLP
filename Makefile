
CXX := gcc


DEBUG ?= 1

ifeq ($(DEBUG),1)
	CXXFLAGS := -g -O0
else
	CXXFLAGS := -s -O3 -march=native -ffast-math
endif

CXXFLAGS += -Wall -Wextra -std=c99 -Iheaders -MMD -MP -fopenmp

LDFLAGS := -lm -lglut -lGLU -lGL -fopenmp

BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj
TARGET    := $(BUILD_DIR)/main

SRC  := main.c $(wildcard src/*.c)

OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(SRC))
DEPS := $(OBJS:.o=.d)

all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Compile
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.aig: %.aag
	~/aiger/aigtoaig $< > $@

%.dot: %.aig
	~/aiger/aigtodot $< > $@

%.png: %.dot
	dot -Tpng $< -o $@

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

-include $(DEPS)

.PHONY: all clean run