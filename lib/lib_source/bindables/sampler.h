#pragma once

#include "../bindable.h"

class Sampler : public Bindable{
    public:
        Sampler(GraphicsEngine& gfx, D3D11_FILTER filter);
        void bind(const GraphicsEngine& gfx) override;
    protected:
	    Microsoft::WRL::ComPtr<ID3D11SamplerState> pSampler;
};