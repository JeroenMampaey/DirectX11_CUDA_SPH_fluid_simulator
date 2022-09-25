CPP_SOURCES = $(wildcard *.cpp)
current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

all: ../lib/binaries/2D_simulation_maker_run.exe
	@powershell '(Write-Host "Executable build successfully, the executable can be found in the binaries folder in the lib directory" -ForegroundColor green)'

../lib/binaries/2D_simulation_maker_run.exe: ../lib/binaries/graphics.dll main.cpp
	nvcc ${CPP_SOURCES} -l ../lib/binaries/graphics -o ../lib/binaries/2D_simulation_maker_run.exe -D SIMULATION_LAYOUT_DIRECTORY='"${current_dir}../simulation_layout/"' -std=c++17

../lib/binaries/graphics.dll:
	@powershell '(Write-Host "The Graphics libary has not been build yet" -ForegroundColor red)'
	@powershell '(Write-Host "Go to the lib folder and use the Makefile to build the library first" -ForegroundColor red)'
	@exit 1

clean:
	@powershell '(Remove-Item -ErrorAction Ignore ../lib/binaries/2D_simulation_maker_run.exe)'