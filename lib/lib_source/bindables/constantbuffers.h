#pragma once

#include "bindable.h"

template<typename C>
class ConstantBuffer : public Bindable{
    public:
        void update(const C& consts) const{
			HRESULT hr;
			D3D11_MAPPED_SUBRESOURCE msr;
			GFX_THROW_FAILED(helper->getContext().Map(pConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &msr));
			memcpy(msr.pData, &consts, sizeof(consts));
			helper->getContext().Unmap(pConstantBuffer.Get(), 0);
		}
        ConstantBuffer(std::shared_ptr<BindableHelper> pHelper, const C& consts)
			:
			Bindable(std::move(pHelper))
		{
			HRESULT hr;
			D3D11_BUFFER_DESC cbd;
			cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			cbd.Usage = D3D11_USAGE_DYNAMIC;
			cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			cbd.MiscFlags = 0;
			cbd.ByteWidth = sizeof(consts);
			cbd.StructureByteStride = 0;

			D3D11_SUBRESOURCE_DATA csd = {};
			csd.pSysMem = &consts;
			GFX_THROW_FAILED(helper->getDevice().CreateBuffer(&cbd, &csd, &pConstantBuffer));
		}
        ConstantBuffer(std::shared_ptr<BindableHelper> pHelper)
			:
			Bindable(std::move(pHelper))
		{
			HRESULT hr;
			D3D11_BUFFER_DESC cbd;
			cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			cbd.Usage = D3D11_USAGE_DYNAMIC;
			cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			cbd.MiscFlags = 0;
			cbd.ByteWidth = sizeof(C);
			cbd.StructureByteStride = 0;
			GFX_THROW_FAILED(helper->getDevice().CreateBuffer(&cbd, nullptr, &pConstantBuffer));
		}
		virtual ~ConstantBuffer() = default;
    protected:
        Microsoft::WRL::ComPtr<ID3D11Buffer> pConstantBuffer;
};

template<typename C>
class VertexConstantBuffer : public ConstantBuffer<C>{
        using ConstantBuffer<C>::pConstantBuffer;
    public:
        using ConstantBuffer<C>::ConstantBuffer;
		VertexConstantBuffer(std::shared_ptr<BindableHelper> pHelper, UINT startSlot, const C& consts)
			:
			ConstantBuffer<C>(std::move(pHelper), consts),
			startSlot(startSlot)
		{}
		VertexConstantBuffer(std::shared_ptr<BindableHelper> pHelper, UINT startSlot)
			:
			ConstantBuffer<C>(std::move(pHelper)),
			startSlot(startSlot)
		{}
        void bind() const override{
            helper->getContext().VSSetConstantBuffers(startSlot, 1, pConstantBuffer.GetAddressOf());
        }
	private:
		UINT startSlot;
};

template<typename C>
class PixelConstantBuffer : public ConstantBuffer<C>{
        using ConstantBuffer<C>::pConstantBuffer;
    public:
        using ConstantBuffer<C>::ConstantBuffer;
		PixelConstantBuffer(std::shared_ptr<BindableHelper> pHelper, UINT startSlot, const C& consts)
			:
			ConstantBuffer<C>(std::move(pHelper), consts),
			startSlot(startSlot)
		{}
		PixelConstantBuffer(std::shared_ptr<BindableHelper> pHelper, UINT startSlot)
			:
			ConstantBuffer<C>(std::move(pHelper)),
			startSlot(startSlot)
		{}
        void bind() const override{
            helper->getContext().PSSetConstantBuffers(0, 1, pConstantBuffer.GetAddressOf());
        }
	private:
		UINT startSlot;
};