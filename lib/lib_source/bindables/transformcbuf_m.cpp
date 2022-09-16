#include "transformcbuf_m.h"
#include "../graphics_engine.h"

std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> TransformCbufM::pVcbuf = nullptr;

TransformCbufM::TransformCbufM(GraphicsEngine& gfx, const Drawable& parent) : parent(parent){
    if(!pVcbuf){
        pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, gfx);
    }
}

void TransformCbufM::bind(const GraphicsEngine& gfx){
    pVcbuf->update(gfx,
        DirectX::XMMatrixTranspose(
            parent.getModel()
        )
    );
    pVcbuf->bind(gfx);
}