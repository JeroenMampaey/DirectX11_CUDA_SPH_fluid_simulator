#pragma once

#include <wrl/client.h>
#include "../bindable.h"

class Sampler : public Bindable{
    public:
        Sampler(GraphicsEngine& gfx);
        void bind(GraphicsEngine& gfx, DrawableState& drawableState) override;
    protected:
	    Microsoft::WRL::ComPtr<ID3D11SamplerState> pSampler;
};