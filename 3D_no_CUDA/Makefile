all: run.exe

run.exe: main.cpp physics.h
	nvcc main.cpp -o run.exe

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"