#include "sampler.h"

Sampler::Sampler(GraphicsEngine& gfx, D3D11_FILTER filter){
	HRESULT hr;

	D3D11_SAMPLER_DESC samplerDesc = {};
	samplerDesc.Filter = filter;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

    GFX_THROW_FAILED(getDevice(gfx)->CreateSamplerState(&samplerDesc, &pSampler));
}

void Sampler::bind(const GraphicsEngine& gfx){
    getContext(gfx)->PSSetSamplers(0, 1, pSampler.GetAddressOf());
}