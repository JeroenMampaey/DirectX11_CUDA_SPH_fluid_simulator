CPP_SOURCES = $(wildcard *.cpp *.cu)
current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
NVCC_PATH := $(shell where nvcc 2> nul)

all: ../lib/binaries/2D_with_CUDA_run.exe
	@powershell '(Write-Host "Executable build successfully, the executable can be found in the binaries folder in the lib directory" -ForegroundColor green)'

../lib/binaries/2D_with_CUDA_run.exe: ../lib/binaries/graphics.dll main.cpp
ifdef NVCC_PATH
	nvcc ${CPP_SOURCES} -l ../lib/binaries/graphics -o ../lib/binaries/2D_with_CUDA_run.exe -D SIMULATION_LAYOUT_DIRECTORY='"${current_dir}../simulation_layout/"' -std=c++17
else
	@powershell '(Write-Host "This program requires the NVCC compiler but it could not be found" -ForegroundColor red)'
	@powershell '(Write-Host "If you want to be able to run this program, download the NVCC compiler" -ForegroundColor red)'
	@exit 1
endif

../lib/binaries/graphics.dll:
	@powershell '(Write-Host "The Graphics libary has not been build yet" -ForegroundColor red)'
	@powershell '(Write-Host "Go to the lib folder and use the Makefile to build the library first" -ForegroundColor red)'
	@exit 1

clean:
	@powershell '(Remove-Item -ErrorAction Ignore ../lib/binaries/2D_with_CUDA_run.exe)'
	@powershell '(Remove-Item -ErrorAction Ignore ../lib/binaries/2D_with_CUDA_run.exp)'
	@powershell '(Remove-Item -ErrorAction Ignore ../lib/binaries/2D_with_CUDA_run.lib)'