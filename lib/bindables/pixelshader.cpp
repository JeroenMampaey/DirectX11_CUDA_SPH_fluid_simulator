#include "pixelshader.h"

PixelShader::PixelShader(GraphicsEngine& gfx, const std::wstring& path){
    Microsoft::WRL::ComPtr<ID3DBlob> pBlob;
    HRESULT hr;
    GFX_THROW_FAILED(D3DReadFileToBlob(path.c_str(), &pBlob));
	GFX_THROW_FAILED(GetDevice(gfx)->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &pPixelShader));
}

void PixelShader::Bind(GraphicsEngine& gfx){
	GetContext(gfx)->PSSetShader(pPixelShader.Get(), nullptr, 0);
}