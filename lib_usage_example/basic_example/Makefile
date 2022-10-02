CPP_SOURCES = $(wildcard *.cpp)

all: ../../lib/binaries/basic_example_run.exe
	@powershell '(Write-Host "Executable build successfully, the executable can be found in the binaries folder in the lib directory" -ForegroundColor green)'

../../lib/binaries/basic_example_run.exe: ../../lib/binaries/graphics.dll main.cpp
	nvcc ${CPP_SOURCES} -l ../../lib/binaries/graphics -o ../../lib/binaries/basic_example_run.exe -std=c++17

../../lib/binaries/graphics.dll:
	@powershell '(Write-Host "The Graphics libary has not been build yet" -ForegroundColor red)'
	@powershell '(Write-Host "Go to the lib folder and use the Makefile to build the library first" -ForegroundColor red)'
	@exit 1

clean:
	@powershell '(Remove-Item -ErrorAction Ignore ../../lib/binaries/basic_example_run.exe)'