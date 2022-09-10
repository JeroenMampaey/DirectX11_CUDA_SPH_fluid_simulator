#pragma once

#include "../bindable.h"
#include <vector>
#include <wrl/client.h>

class VertexBuffer : public Bindable{
    public:
        template<class V>
        VertexBuffer(GraphicsEngine& gfx, const std::vector<V>& vertices) : stride(sizeof(V)){
            HRESULT hr;
            D3D11_BUFFER_DESC bd = {};
            bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.CPUAccessFlags = 0;
            bd.MiscFlags = 0;
            bd.ByteWidth = UINT(sizeof(V)*vertices.size());
            bd.StructureByteStride = sizeof(V);
            D3D11_SUBRESOURCE_DATA sd = {};
            sd.pSysMem = vertices.data();
            GFX_THROW_FAILED(getDevice(gfx)->CreateBuffer(&bd, &sd, &pVertexBuffer));
        }
        void bind(GraphicsEngine& gfx, DrawableState& drawableState) override;
    protected:
        UINT stride;
        Microsoft::WRL::ComPtr<ID3D11Buffer> pVertexBuffer;
};