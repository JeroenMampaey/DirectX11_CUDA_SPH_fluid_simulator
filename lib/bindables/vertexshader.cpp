#include "vertexshader.h"


VertexShader::VertexShader(GraphicsEngine& gfx,const std::wstring& path){
    HRESULT hr;
	GFX_THROW_FAILED(D3DReadFileToBlob(path.c_str(), &pBytecodeBlob));
	GFX_THROW_FAILED(GetDevice(gfx)->CreateVertexShader( 
		pBytecodeBlob->GetBufferPointer(),
		pBytecodeBlob->GetBufferSize(),
		nullptr,
		&pVertexShader 
	));
}

void VertexShader::Bind(GraphicsEngine& gfx){
	GetContext(gfx)->VSSetShader(pVertexShader.Get(), nullptr, 0);
}

ID3DBlob* VertexShader::GetBytecode() const noexcept{
	return pBytecodeBlob.Get();
}