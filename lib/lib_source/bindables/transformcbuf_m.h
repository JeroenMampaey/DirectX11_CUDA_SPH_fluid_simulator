#pragma once

#include "constantbuffers.h"
#include "../drawable.h"
#include "../bindable.h"

class TransformCbufM : public Bindable{
    public:
        TransformCbufM(GraphicsEngine& gfx, const Drawable& parent);
        void bind(const GraphicsEngine& gfx) override;
    private:
        static std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
        const Drawable& parent;
};