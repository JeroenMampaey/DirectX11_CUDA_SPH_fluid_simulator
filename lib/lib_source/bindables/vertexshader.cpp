#include "vertexshader.h"


VertexShader::VertexShader(std::shared_ptr<BindableHelper> helper, const std::wstring& path)
	:
	Bindable(helper)
{
    HRESULT hr;
	GFX_THROW_FAILED(D3DReadFileToBlob(path.c_str(), &pBytecodeBlob));
	GFX_THROW_FAILED(helper->getDevice().CreateVertexShader( 
		pBytecodeBlob->GetBufferPointer(),
		pBytecodeBlob->GetBufferSize(),
		nullptr,
		&pVertexShader 
	));
}

void VertexShader::bind(){
	helper->getContext().VSSetShader(pVertexShader.Get(), nullptr, 0);
}

ID3DBlob* VertexShader::getBytecode() const noexcept{
	return pBytecodeBlob.Get();
}