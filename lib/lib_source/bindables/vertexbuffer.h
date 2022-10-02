#pragma once

#include "bindable.h"
#include <vector>

class VertexBuffer : public Bindable{
    public:
        virtual ~VertexBuffer() noexcept = default;
        void bind() const override;

    protected:
        VertexBuffer(std::shared_ptr<BindableHelper> pHelper, const void* vertexBuffers[], const size_t vertexSizes[], const UINT cpuAccessFlags[], const size_t numVertices[], const int numVertexBuffers);
        
        std::vector<Microsoft::WRL::ComPtr<ID3D11Buffer>> vertexBufferPs;

        std::vector<UINT> strides;
        std::vector<UINT> offsets;
        std::vector<ID3D11Buffer*> preparedBuffers;
};

class ConstantVertexBuffer : public VertexBuffer{
    public:
        ConstantVertexBuffer(std::shared_ptr<BindableHelper> pHelper, const void* vertexBuffers[], const size_t vertexSizes[], const size_t numVertices[], const int numVertexBuffers);
};

class MappableVertexBuffer : public VertexBuffer{
        using VertexBuffer::VertexBuffer;
    public:
        virtual ~MappableVertexBuffer() noexcept = default;
        virtual void* getMappedAccess(int vertexBufferIndex) const = 0;
        virtual void unMap(int vertexBufferIndex) const noexcept = 0;
};

class CpuMappableVertexBuffer : public MappableVertexBuffer{
    public:
        CpuMappableVertexBuffer(std::shared_ptr<BindableHelper> pHelper, const void* vertexBuffers[], const size_t vertexSizes[], const UINT cpuAccessFlags[], const size_t numVertices[], const int numVertexBuffers);
        void* getMappedAccess(int vertexBufferIndex) const override;
        void unMap(int vertexBufferIndex) const noexcept override;
};