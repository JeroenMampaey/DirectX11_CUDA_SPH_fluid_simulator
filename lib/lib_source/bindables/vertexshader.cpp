#include "vertexshader.h"


VertexShader::VertexShader(GraphicsEngine& gfx,const std::wstring& path){
    HRESULT hr;
	GFX_THROW_FAILED(D3DReadFileToBlob(path.c_str(), &pBytecodeBlob));
	GFX_THROW_FAILED(getDevice(gfx)->CreateVertexShader( 
		pBytecodeBlob->GetBufferPointer(),
		pBytecodeBlob->GetBufferSize(),
		nullptr,
		&pVertexShader 
	));
}

void VertexShader::bind(const GraphicsEngine& gfx, DrawableState& drawableState){
	getContext(gfx)->VSSetShader(pVertexShader.Get(), nullptr, 0);
}

ID3DBlob* VertexShader::getBytecode() const noexcept{
	return pBytecodeBlob.Get();
}