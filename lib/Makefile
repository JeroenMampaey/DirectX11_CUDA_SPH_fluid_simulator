CPP_SOURCES = $(wildcard lib_source/*.cpp lib_source/bindables/*.cpp lib_source/drawers/*.cpp lib_source/events/*.cpp)
PIXELSHADERS = $(patsubst lib_source/pixelshaders/%.hlsl, binaries/pixelshaders/%.cso, $(wildcard lib_source/pixelshaders/*.hlsl))
VERTEXSHADERS = $(patsubst lib_source/vertexshaders/%.hlsl, binaries/vertexshaders/%.cso, $(wildcard lib_source/vertexshaders/*.hlsl))
current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
NVCC_PATH := $(shell where nvcc 2> nul)
CL_PATH := $(shell where cl 2> nul)

all: ${PIXELSHADERS} ${VERTEXSHADERS} binaries/graphics.dll
	@powershell '(Write-Host "Library build successfully" -ForegroundColor green)'

binaries/graphics.dll:
ifdef NVCC_PATH
	nvcc -o binaries/graphics.dll --shared ${CPP_SOURCES} -D DEFAULT_VERTEX_SHADERS_DIRECTORY='L"${current_dir}binaries/vertexshaders/"' -D DEFAULT_PIXEL_SHADERS_DIRECTORY='L"${current_dir}binaries/pixelshaders/"' -std=c++17
else 
ifdef CL_PATH
	@powershell '(Write-Host "Warning: the graphics library is being compiled without CUDA support because the NVCC compiler could not be found" -ForegroundColor DarkYellow)'
	@powershell '(Write-Host "If you want to enable CUDA support, download the NVCC compiler" -ForegroundColor DarkYellow)'
	@powershell '(Write-Host "" -ForegroundColor DarkYellow)'
	cl.exe /Fe:binaries/graphics.dll /LD ${CPP_SOURCES} -D DEFAULT_VERTEX_SHADERS_DIRECTORY='L"${current_dir}binaries/vertexshaders/"' -D DEFAULT_PIXEL_SHADERS_DIRECTORY='L"${current_dir}binaries/pixelshaders/"' /std:c++17 /EHsc /link /DLL
	@powershell '(del *.obj)'
else
	@powershell '(Write-Host "Could not find an MSVC compiler, for this library either NVCC or the cl.exe tool are required" -ForegroundColor red)'
	@exit 1
endif
endif

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