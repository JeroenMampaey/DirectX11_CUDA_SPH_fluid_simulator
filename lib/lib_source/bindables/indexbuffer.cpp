#include "indexbuffer.h"

IndexBuffer::IndexBuffer(GraphicsEngine& gfx, const std::vector<unsigned short>& indices){
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
	GFX_THROW_FAILED(getDevice(gfx)->CreateBuffer(&ibd, &isd, &pIndexBuffer));
}

void IndexBuffer::bind(GraphicsEngine& gfx, DrawableState& DrawableState){
    getContext(gfx)->IASetIndexBuffer(pIndexBuffer.Get(), DXGI_FORMAT_R16_UINT, 0);
}