CPP_SOURCES = $(wildcard *.cpp ../lib/*.cpp)

all: run.exe

run.exe: main.cpp
	nvcc ${CPP_SOURCES} -o run.exe -std=c++17

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"