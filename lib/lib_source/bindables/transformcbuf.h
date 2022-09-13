#pragma once

#include "constantbuffers.h"
#include "../drawable.h"
#include "../bindable.h"
#include <DirectXMath.h>

class TransformCbuf : public Bindable{
    public:
        TransformCbuf(GraphicsEngine& gfx);
        void bind(const GraphicsEngine& gfx, DrawableState& drawableState) override;
    private:
        static std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};