all: run.exe

run.exe: main.cu physics.cu ../extra_code/console_debug.cpp ../extra_code/basewin.h physics.h ../extra_code/console_debug.h
	nvcc main.cu physics.cu ../extra_code/console_debug.cpp -o run.exe --resource-usage

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"
	powershell "(Remove-Item -ErrorAction Ignore run.exp)"
	powershell "(Remove-Item -ErrorAction Ignore run.lib)"