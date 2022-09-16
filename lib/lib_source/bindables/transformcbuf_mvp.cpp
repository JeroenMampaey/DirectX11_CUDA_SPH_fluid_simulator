#include "transformcbuf_mvp.h"
#include "../graphics_engine.h"

std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> TransformCbufMVP::pVcbuf = nullptr;

TransformCbufMVP::TransformCbufMVP(GraphicsEngine& gfx, const Drawable& parent) : parent(parent){
    if(!pVcbuf){
        pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, gfx);
    }
}

void TransformCbufMVP::bind(const GraphicsEngine& gfx){
    pVcbuf->update(gfx,
        DirectX::XMMatrixTranspose(
            parent.getModel() * gfx.getView() * gfx.getProjection()
        )
    );
    pVcbuf->bind(gfx);
}