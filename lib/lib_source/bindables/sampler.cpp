#include "sampler.h"

Sampler::Sampler(std::shared_ptr<BindableHelper> pHelper, D3D11_FILTER filter)
	:
	Bindable(std::move(pHelper))
{
	HRESULT hr;

	D3D11_SAMPLER_DESC samplerDesc = {};
	samplerDesc.Filter = filter;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

    GFX_THROW_FAILED(helper->getDevice().CreateSamplerState(&samplerDesc, &pSampler));
}

void Sampler::bind() const{
    helper->getContext().PSSetSamplers(0, 1, pSampler.GetAddressOf());
}