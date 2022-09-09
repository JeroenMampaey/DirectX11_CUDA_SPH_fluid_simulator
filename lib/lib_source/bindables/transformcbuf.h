#pragma once

#include "constantbuffers.h"
#include "../drawable.h"
#include "../bindable.h"
#include <DirectXMath.h>

class TransformCbuf : public Bindable
{
    public:
        TransformCbuf(GraphicsEngine& gfx){
            if(!pVcbuf){
                pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(gfx);
            }
        }
        void Bind(GraphicsEngine& gfx, DrawableState& drawableState) override{
            pVcbuf->Update(gfx,
                DirectX::XMMatrixTranspose(
                    drawableState.getTransformXM() * gfx.GetProjection()
                )
            );
            pVcbuf->Bind(gfx, drawableState);
        }
    private:
        static std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};