CPP_SOURCES = $(wildcard *.cpp)
NVCC_PATH := $(shell where nvcc 2> nul)
CL_PATH := $(shell where cl 2> nul)

all: ../../lib/binaries/instanced_drawing_example_run.exe
	@powershell '(Write-Host "Executable build successfully, the executable can be found in the binaries folder in the lib directory" -ForegroundColor green)'

../../lib/binaries/instanced_drawing_example_run.exe: ../../lib/binaries/graphics.dll main.cpp
ifdef NVCC_PATH
	nvcc ${CPP_SOURCES} -l ../../lib/binaries/graphics -o ../../lib/binaries/instanced_drawing_example_run.exe -std=c++17
else 
ifdef CL_PATH
	cl.exe /Fe:../../lib/binaries/instanced_drawing_example_run.exe ${CPP_SOURCES} ../../lib/binaries/graphics.lib /std:c++17 /EHsc
	@powershell '(del *.obj)'
else
	@powershell '(Write-Host "Could not find an MSVC compiler, for this library either NVCC or the cl.exe tool are required" -ForegroundColor red)'
	@exit 1
endif
endif

../../lib/binaries/graphics.dll:
	@powershell '(Write-Host "The Graphics libary has not been build yet" -ForegroundColor red)'
	@powershell '(Write-Host "Go to the lib folder and use the Makefile to build the library first" -ForegroundColor red)'
	@exit 1

clean:
	@powershell '(Remove-Item -ErrorAction Ignore ../../lib/binaries/instanced_drawing_example_run.exe)'