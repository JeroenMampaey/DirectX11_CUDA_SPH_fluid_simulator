#include "vertexbuffer.h"
#include <array>

VertexBuffer::VertexBuffer(std::shared_ptr<BindableHelper> helper, const void* vertexBuffers[], const size_t vertexSizes[], const UINT cpuAccessFlags[], const size_t numVertices[], const int numVertexBuffers)
	:
	Bindable(helper),
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
		bd.Usage = cpuAccessFlags[i]==0 ? D3D11_USAGE_DEFAULT : D3D11_USAGE_DYNAMIC;
		bd.CPUAccessFlags = cpuAccessFlags[i];
		//bd.CPUAccessFlags = cpuAccessFlags[i] ? D3D11_CPU_ACCESS_WRITE : 0;
		bd.MiscFlags = 0;
		bd.ByteWidth = UINT(vertexSizes[i]*numVertices[i]);
		bd.StructureByteStride = vertexSizes[i];

		D3D11_SUBRESOURCE_DATA sd = {};
		sd.pSysMem = vertexBuffers[i];
		GFX_THROW_FAILED(helper->getDevice().CreateBuffer(&bd, &sd, &vertexBufferPs[i]));

		strides[i] = vertexSizes[i];
		offsets[i] = 0;
		preparedBuffers[i] = vertexBufferPs[i].Get();
	}
}

void VertexBuffer::bind(){
	helper->getContext().IASetVertexBuffers(0, vertexBufferPs.size(), preparedBuffers.data(), strides.data(), offsets.data());
}

ConstantVertexBuffer::ConstantVertexBuffer(std::shared_ptr<BindableHelper> helper, const void* vertexBuffers[], const size_t vertexSizes[], const size_t numVertices[], const int numVertexBuffers)
	:
	VertexBuffer(helper, vertexBuffers, vertexSizes, std::vector<UINT>(numVertexBuffers, 0).data(), numVertices, numVertexBuffers)
{}

CpuMappableVertexBuffer::CpuMappableVertexBuffer(std::shared_ptr<BindableHelper> helper, const void* vertexBuffers[], const size_t vertexSizes[], const UINT cpuAccessFlags[], const size_t numVertices[], const int numVertexBuffers)
	:
	MappableVertexBuffer(helper, vertexBuffers, vertexSizes, cpuAccessFlags, numVertices, numVertexBuffers)
{}

void* CpuMappableVertexBuffer::getMappedAccess(int vertexBufferIndex) const{
	D3D11_BUFFER_DESC desc;
	vertexBufferPs[vertexBufferIndex]->GetDesc(&desc);

	if(desc.CPUAccessFlags != D3D11_CPU_ACCESS_WRITE){
		throw std::exception("Attempted to map a vertexbuffer that is not accessible from the CPU.");
	}

	HRESULT hr;
	D3D11_MAPPED_SUBRESOURCE msr;
	GFX_THROW_FAILED(helper->getContext().Map(vertexBufferPs[vertexBufferIndex].Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &msr));
	return msr.pData;
}

void CpuMappableVertexBuffer::unMap(int vertexBufferIndex) const noexcept{
	helper->getContext().Unmap(vertexBufferPs[vertexBufferIndex].Get(), 0);
}