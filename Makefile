CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -mavx -mavx2 -mfma -mavx512f -mavx512vl
TARGET := gpt_mini
SRC := main.cpp gpt_mini.cpp

.PHONY: all clean run verbose

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

run: $(TARGET)
ifeq ($(filter verbose,$(MAKECMDGOALS)),verbose)
	./$(TARGET) -verbose
else
	./$(TARGET)
endif

verbose:
	@:

clean:
	rm -rf $(TARGET) $(EXPORT_TARGET) gpt_mini_baseline output