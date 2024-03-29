#include "indexbuffer.h"

IndexBuffer::IndexBuffer(std::shared_ptr<BindableHelper> pHelper, const std::vector<unsigned short>& indices)
	:
	Bindable(std::move(pHelper))
{
    HRESULT hr;
    D3D11_BUFFER_DESC ibd = {};
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.Usage = D3D11_USAGE_DEFAULT;
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.ByteWidth = UINT(indices.size()*sizeof(unsigned short));
	ibd.StructureByteStride = sizeof(unsigned short);
	D3D11_SUBRESOURCE_DATA isd = {};
	isd.pSysMem = indices.data();
	GFX_THROW_FAILED(helper->getDevice().CreateBuffer(&ibd, &isd, &pIndexBuffer));
}

void IndexBuffer::bind() const{
    helper->getContext().IASetIndexBuffer(pIndexBuffer.Get(), DXGI_FORMAT_R16_UINT, 0);
}