INCDIR := include

CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++20 -I$(INCDIR)
LDFLAGS :=
SRCDIR := src
BINDIR := bin

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.o,$(SOURCES))

TARGET := $(BINDIR)/app

.PHONY: all clean run dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BINDIR)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile .cpp -> .o
$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(BINDIR)/*.o $(TARGET)

run: all
	./$(TARGET)
