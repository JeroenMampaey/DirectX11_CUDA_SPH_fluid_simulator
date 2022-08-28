all: run.exe

run.exe: main.cpp graphics.cpp physics.h graphics.h ../extra_code/basewin.h
	nvcc main.cpp graphics.cpp -o run.exe

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"