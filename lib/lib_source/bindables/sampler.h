#pragma once

#include "bindable.h"

class Sampler : public Bindable{
    public:
        Sampler(std::shared_ptr<BindableHelper> pHelper, D3D11_FILTER filter);
        void bind() const override;
    protected:
	    Microsoft::WRL::ComPtr<ID3D11SamplerState> pSampler;
};