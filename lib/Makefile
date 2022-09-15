CPP_SOURCES = $(wildcard lib_source/*.cpp lib_source/bindables/*.cpp lib_source/drawables/*.cpp lib_source/events/*.cpp)
PIXELSHADERS = $(patsubst lib_source/pixelshaders/%.hlsl, binaries/pixelshaders/%.cso, $(wildcard lib_source/pixelshaders/*.hlsl))
VERTEXSHADERS = $(patsubst lib_source/vertexshaders/%.hlsl, binaries/vertexshaders/%.cso, $(wildcard lib_source/vertexshaders/*.hlsl))
current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

all: ${PIXELSHADERS} ${VERTEXSHADERS} binaries/graphics.dll
	@powershell '(Write-Host "Library build successfully" -ForegroundColor green)'

binaries/graphics.dll:
	nvcc -o binaries/graphics.dll --shared ${CPP_SOURCES} -D LIBRARY_EXPORTS=1 -D DEFAULT_VERTEX_SHADERS_DIRECTORY='L"${current_dir}binaries/vertexshaders/"' -D DEFAULT_PIXEL_SHADERS_DIRECTORY='L"${current_dir}binaries/pixelshaders/"' -std=c++17

binaries/pixelshaders/%.cso: lib_source/pixelshaders/%.hlsl binaries/pixelshaders
	fxc.exe $< -Fo $@ -T ps_4_1

binaries/pixelshaders:
	@powershell '(mkdir binaries/pixelshaders)'

binaries/vertexshaders/%.cso: lib_source/vertexshaders/%.hlsl binaries/vertexshaders
	fxc.exe $< -Fo $@ -T vs_4_1

binaries/vertexshaders:
	@powershell '(mkdir binaries/vertexshaders)'

clean:
	@powershell '(Remove-Item -ErrorAction Ignore binaries/vertexshaders/*.cso)'
	@powershell '(Remove-Item -ErrorAction Ignore binaries/pixelshaders/*.cso)'
	@powershell '(Remove-Item -ErrorAction Ignore binaries/graphics.dll)'
	@powershell '(Remove-Item -ErrorAction Ignore binaries/graphics.exp)'
	@powershell '(Remove-Item -ErrorAction Ignore binaries/graphics.lib)'