CPP_SOURCES = $(wildcard *.cpp ../lib/*.cpp ../lib/bindables/*.cpp ../lib/drawables/*.cpp)

all: run.exe

run.exe: main.cpp VertexShader.cso PixelShader.cso
	nvcc ${CPP_SOURCES} -o run.exe -std=c++17

VertexShader.cso: VertexShader.hlsl
	fxc.exe VertexShader.hlsl -Fo VertexShader.cso -T vs_4_1

PixelShader.cso: PixelShader.hlsl
	fxc.exe PixelShader.hlsl -Fo PixelShader.cso -T ps_4_1

clean:
	powershell "(Remove-Item -ErrorAction Ignore VertexShader.cso)"
	powershell "(Remove-Item -ErrorAction Ignore PixelShader.cso)"
	powershell "(Remove-Item -ErrorAction Ignore run.exe)"