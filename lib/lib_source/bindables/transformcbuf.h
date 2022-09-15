#pragma once

#include "constantbuffers.h"
#include "../drawable.h"
#include "../bindable.h"

class TransformCbuf : public Bindable{
    public:
        TransformCbuf(GraphicsEngine& gfx, const Drawable& parent);
        void bind(const GraphicsEngine& gfx) override;
    private:
        static std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
        const Drawable& parent;
};