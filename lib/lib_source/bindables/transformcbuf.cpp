#include "transformcbuf.h"

std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> TransformCbuf::pVcbuf = nullptr;

TransformCbuf::TransformCbuf(GraphicsEngine& gfx){
    if(!pVcbuf){
        pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(gfx);
    }
}

void TransformCbuf::bind(const GraphicsEngine& gfx, DrawableState& drawableState){
    pVcbuf->update(gfx,
        DirectX::XMMatrixTranspose(
            drawableState.getTransformXM() * gfx.getProjection()
        )
    );
    pVcbuf->bind(gfx, drawableState);
}