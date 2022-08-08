all: run.exe

run.exe: main.cpp physics.cpp ../extra_code/console_debug.cpp ../extra_code/basewin.h physics.h boundary.h particle.h neighbor.h pump.h ../extra_code/console_debug.h
	nvcc main.cpp physics.cpp ../extra_code/console_debug.cpp -o run.exe

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"