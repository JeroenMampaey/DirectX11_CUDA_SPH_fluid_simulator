all: run.exe

run.exe: main.cpp physics.cpp basewin.h physics.h boundary.h particle.h
	nvcc main.cpp physics.cpp -o run.exe

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"