all: run.exe

run.exe: main.cpp simulation_builder.cpp ../basewin.h
	nvcc main.cpp simulation_builder.cpp -o run.exe

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"
	powershell "(Remove-Item -ErrorAction Ignore simulation.txt)"