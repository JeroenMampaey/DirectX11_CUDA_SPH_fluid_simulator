#include "vertexbuffer.h"

VertexBuffer::VertexBuffer(GraphicsEngine& gfx, const void* vertexBuffers[], const size_t vertexSizes[], const bool cpuAccessFlags[], const int numVertexBuffers, const size_t numVertices)
	:
	numVertices(numVertices),
	vertexBufferPs(numVertexBuffers),
	strides(numVertexBuffers),
	offsets(numVertexBuffers),
	preparedBuffers(numVertexBuffers)
{
	HRESULT hr;
	vertexBufferPs = std::vector<Microsoft::WRL::ComPtr<ID3D11Buffer>>(numVertexBuffers);
	strides = std::vector<UINT>(numVertexBuffers);
	for(int i=0; i<numVertexBuffers; i++){
		D3D11_BUFFER_DESC bd = {};
		bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		bd.Usage = cpuAccessFlags[i] ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
		bd.CPUAccessFlags = cpuAccessFlags[i] ? D3D11_CPU_ACCESS_WRITE : 0;
		bd.MiscFlags = 0;
		bd.ByteWidth = UINT(vertexSizes[i]*numVertices);
		bd.StructureByteStride = vertexSizes[i];

		D3D11_SUBRESOURCE_DATA sd = {};
		sd.pSysMem = vertexBuffers[i];
		GFX_THROW_FAILED(getDevice(gfx)->CreateBuffer(&bd, &sd, &vertexBufferPs[i]));

		strides[i] = vertexSizes[i];
		offsets[i] = 0;
		preparedBuffers[i] = vertexBufferPs[i].Get();
	}
}

void VertexBuffer::bind(const GraphicsEngine& gfx){
	getContext(gfx)->IASetVertexBuffers(0, vertexBufferPs.size(), preparedBuffers.data(), strides.data(), offsets.data());
}

void VertexBuffer::update(const GraphicsEngine& gfx, int vertexBufferIndex, void* newData, size_t size){
	D3D11_BUFFER_DESC desc;
	vertexBufferPs[vertexBufferIndex]->GetDesc(&desc);
	if(desc.ByteWidth < size){
		throw std::exception("Attempted to update a vertexbuffer with new data that was too big");
	}

	if(desc.CPUAccessFlags != D3D11_CPU_ACCESS_WRITE){
		throw std::exception("Attempted to update a vertexbuffer that is not accessible from the CPU");
	}

	HRESULT hr;
	D3D11_MAPPED_SUBRESOURCE msr;
	GFX_THROW_FAILED(getContext(gfx)->Map(vertexBufferPs[vertexBufferIndex].Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &msr));
	memcpy(msr.pData, newData, size);
	getContext(gfx)->Unmap(vertexBufferPs[vertexBufferIndex].Get(), 0);
}