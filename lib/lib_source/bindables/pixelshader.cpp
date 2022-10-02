#include "pixelshader.h"

PixelShader::PixelShader(std::shared_ptr<BindableHelper> pHelper, const std::wstring& path)
    :
    Bindable(std::move(pHelper))
{
    Microsoft::WRL::ComPtr<ID3DBlob> pBlob;
    HRESULT hr;
    GFX_THROW_FAILED(D3DReadFileToBlob(path.c_str(), &pBlob));
	GFX_THROW_FAILED(helper->getDevice().CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &pPixelShader));
}

void PixelShader::bind() const{
	helper->getContext().PSSetShader(pPixelShader.Get(), nullptr, 0);
}