#pragma once

#include "../bindable.h"

template<typename C>
class ConstantBuffer : public Bindable{
    public:
        void update(const GraphicsEngine& gfx, const C& consts){
			HRESULT hr;
			D3D11_MAPPED_SUBRESOURCE msr;
			GFX_THROW_FAILED(getContext(gfx)->Map(pConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &msr));
			memcpy(msr.pData, &consts, sizeof(consts));
			getContext(gfx)->Unmap(pConstantBuffer.Get(), 0);
		}
        ConstantBuffer(GraphicsEngine& gfx, const C& consts){
			HRESULT hr;
			D3D11_BUFFER_DESC cbd;
			cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			cbd.Usage = D3D11_USAGE_DYNAMIC;
			cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			cbd.MiscFlags = 0;
			cbd.ByteWidth = sizeof( consts );
			cbd.StructureByteStride = 0;

			D3D11_SUBRESOURCE_DATA csd = {};
			csd.pSysMem = &consts;
			GFX_THROW_FAILED(getDevice(gfx)->CreateBuffer(&cbd, &csd, &pConstantBuffer));
		}
        ConstantBuffer(GraphicsEngine& gfx){
			HRESULT hr;
			D3D11_BUFFER_DESC cbd;
			cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			cbd.Usage = D3D11_USAGE_DYNAMIC;
			cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			cbd.MiscFlags = 0;
			cbd.ByteWidth = sizeof(C);
			cbd.StructureByteStride = 0;
			GFX_THROW_FAILED(getDevice(gfx)->CreateBuffer(&cbd, nullptr, &pConstantBuffer));
		}
    protected:
        Microsoft::WRL::ComPtr<ID3D11Buffer> pConstantBuffer;
};

template<typename C>
class VertexConstantBuffer : public ConstantBuffer<C>{
        using ConstantBuffer<C>::pConstantBuffer;
        using Bindable::getContext;
    public:
        using ConstantBuffer<C>::ConstantBuffer;
        void bind(const GraphicsEngine& gfx) override{
            getContext(gfx)->VSSetConstantBuffers(0, 1, pConstantBuffer.GetAddressOf());
        }
};

template<typename C>
class PixelConstantBuffer : public ConstantBuffer<C>{
        using ConstantBuffer<C>::pConstantBuffer;
        using Bindable::getContext;
    public:
        using ConstantBuffer<C>::ConstantBuffer;
        void bind(const GraphicsEngine& gfx) override{
            getContext(gfx)->PSSetConstantBuffers(0, 1, pConstantBuffer.GetAddressOf());
        }
};