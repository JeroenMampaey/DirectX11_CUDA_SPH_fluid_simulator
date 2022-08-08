all: run.exe

run.exe: main.cpp simulation_builder.cpp ../extra_code/basewin.h
	nvcc main.cpp simulation_builder.cpp -o run.exe

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"