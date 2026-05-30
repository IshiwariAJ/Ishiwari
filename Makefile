CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c99 -Isrc
LDFLAGS = -lm

LIB_SRC = src/matrix.c \
          src/params.c \
          src/norm.c \
          src/attention.c \
          src/ffn.c \
          src/encoder.c \
          src/decoder.c \
          src/transformer.c \
          src/optimizer.c \
          src/inference.c \
          src/serialize.c \
          src/event.c \
          src/adapters.c \
          src/event_task.c

TARGET      = transformer
TEST_TARGET = test_runner

all: $(TARGET)

$(TARGET): $(LIB_SRC) src/main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST_TARGET): $(LIB_SRC) src/test.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	-cmd /C "del /Q $(TARGET).exe $(TEST_TARGET).exe 2>NUL"
	-cmd /C "del /Q $(TARGET) $(TEST_TARGET) 2>NUL"

.PHONY: all test clean
