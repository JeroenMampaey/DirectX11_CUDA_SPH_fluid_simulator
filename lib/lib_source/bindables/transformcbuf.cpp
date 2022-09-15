#include "transformcbuf.h"
#include "../graphics_engine.h"

std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> TransformCbuf::pVcbuf = nullptr;

TransformCbuf::TransformCbuf(GraphicsEngine& gfx, const Drawable& parent) : parent(parent){
    if(!pVcbuf){
        pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(gfx);
    }
}

void TransformCbuf::bind(const GraphicsEngine& gfx){
    pVcbuf->update(gfx,
        DirectX::XMMatrixTranspose(
            parent.getTransformXM() * gfx.getProjection()
        )
    );
    pVcbuf->bind(gfx);
}