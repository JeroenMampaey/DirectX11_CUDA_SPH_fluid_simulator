#pragma once

#include "constantbuffers.h"
#include "../drawable.h"
#include "../bindable.h"
#include <DirectXMath.h>

template<class T>
class TransformCbuf : public Bindable
{
public:
	TransformCbuf(GraphicsEngine& gfx, const Drawable<T>& parent) : parent(parent) {
        if(!pVcbuf){
            pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(gfx);
        }
    }
	void Bind(GraphicsEngine& gfx) override{
        pVcbuf->Update(gfx,
            DirectX::XMMatrixTranspose(
                parent.GetTransformXM() * gfx.GetProjection()
            )
        );
        pVcbuf->Bind(gfx);
    }
private:
	static std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
	const Drawable<T>& parent;
};

template<class T>
std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> TransformCbuf<T>::pVcbuf = nullptr;