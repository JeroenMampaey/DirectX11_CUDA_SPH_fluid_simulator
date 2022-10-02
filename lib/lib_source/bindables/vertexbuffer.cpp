#include "vertexbuffer.h"
#include <array>
#include "../exceptions.h"

VertexBuffer::VertexBuffer(std::shared_ptr<BindableHelper> pHelper, const void* vertexBuffers[], const size_t vertexSizes[], const UINT cpuAccessFlags[], const size_t numVertices[], const int numVertexBuffers)
	:
	Bindable(std::move(pHelper)),
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

void VertexBuffer::bind() const{
	helper->getContext().IASetVertexBuffers(0, vertexBufferPs.size(), preparedBuffers.data(), strides.data(), offsets.data());
}

ConstantVertexBuffer::ConstantVertexBuffer(std::shared_ptr<BindableHelper> pHelper, const void* vertexBuffers[], const size_t vertexSizes[], const size_t numVertices[], const int numVertexBuffers)
	:
	VertexBuffer(std::move(pHelper), vertexBuffers, vertexSizes, std::vector<UINT>(numVertexBuffers, 0).data(), numVertices, numVertexBuffers)
{}

CpuMappableVertexBuffer::CpuMappableVertexBuffer(std::shared_ptr<BindableHelper> pHelper, const void* vertexBuffers[], const size_t vertexSizes[], const UINT cpuAccessFlags[], const size_t numVertices[], const int numVertexBuffers)
	:
	MappableVertexBuffer(std::move(pHelper), vertexBuffers, vertexSizes, cpuAccessFlags, numVertices, numVertexBuffers)
{}

void* CpuMappableVertexBuffer::getMappedAccess(int vertexBufferIndex){
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

void CpuMappableVertexBuffer::unMap(int vertexBufferIndex){
	D3D11_BUFFER_DESC desc;
	vertexBufferPs[vertexBufferIndex]->GetDesc(&desc);

	if(desc.CPUAccessFlags != D3D11_CPU_ACCESS_WRITE){
		throw std::exception("Attempted to unmap a vertexbuffer that is not accessible from the CPU.");
	}

	helper->getContext().Unmap(vertexBufferPs[vertexBufferIndex].Get(), 0);
}


CudaMappableVertexBuffer::CudaMappableVertexBuffer(std::shared_ptr<BindableHelper> pHelper, const void* vertexBuffers[], const size_t vertexSizes[], const bool cudaAccessibilityMask[], const size_t numVertices[], const int numVertexBuffers)
	:
	MappableVertexBuffer(std::move(pHelper), vertexBuffers, vertexSizes, std::vector<UINT>(numVertexBuffers, 0).data(), numVertices, numVertexBuffers)
{
#if __has_include(<cuda.h>)
	for(int i=0; i<numVertexBuffers; i++){
		cudaResources.push_back(nullptr);
		cudaError_t err;
		if(cudaAccessibilityMask[i]){
			CUDA_THROW_FAILED(cudaGraphicsD3D11RegisterResource(&cudaResources[i], vertexBufferPs[i].Get(), cudaGraphicsRegisterFlagsNone));
		}
	}
#endif
}

void* CudaMappableVertexBuffer::getMappedAccess(int vertexBufferIndex){
#if __has_include(<cuda.h>)
	void* retval;

	cudaGraphicsResource*& cudaResource = cudaResources[vertexBufferIndex];

	if(cudaResource==nullptr){
		throw std::exception("Attempted to map a vertexbuffer that is not accessible to CUDA.");
	}

	cudaError_t err;
	CUDA_THROW_FAILED(cudaGraphicsMapResources(1, &cudaResource, 0));
	size_t num_bytes;
    CUDA_THROW_FAILED(cudaGraphicsResourceGetMappedPointer(&retval, &num_bytes, cudaResource));
	return retval;
#else
	throw std::exception("Trying to get MappedAccess to a CudaMappableVertexBuffer but library 'cuda.h' was not found");
#endif
}

void CudaMappableVertexBuffer::unMap(int vertexBufferIndex){
#if __has_include(<cuda.h>)
	void* retval;

	cudaGraphicsResource*& cudaResource = cudaResources[vertexBufferIndex];

	if(cudaResource==nullptr){
		throw std::exception("Attempted to map a vertexbuffer that is not accessible to CUDA.");
	}

	cudaError_t err;
	CUDA_THROW_FAILED(cudaGraphicsUnmapResources(1, &cudaResource, 0));
#else
	throw std::exception("Trying to unMap a CudaMappableVertexBuffer but library 'cuda.h' was not found");
#endif
}

#if __has_include(<cuda.h>)
CudaMappableVertexBuffer::~CudaMappableVertexBuffer() noexcept{
	for(cudaGraphicsResource*& cudaResource : cudaResources){
		if(cudaResource!=nullptr){
			cudaGraphicsUnregisterResource(cudaResource);
		}
	}
}
#endif