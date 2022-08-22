all: run.exe

run.exe: main.cu physics.cu ../extra_code/console_debug.cpp ../extra_code/basewin.h physics.h ../extra_code/console_debug.h
	nvcc main.cu physics.cu ../extra_code/console_debug.cpp -o run.exe -prec-sqrt=true

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"