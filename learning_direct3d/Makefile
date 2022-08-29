all: run.exe

run.exe: main.cpp graphics.cpp my_exception.cpp physics.h graphics.h ../extra_code/basewin.h my_exception.h
	nvcc main.cpp graphics.cpp my_exception.cpp -o run.exe

clean:
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"