#pragma once

#include "../bindable.h"
#include <vector>

class VertexBuffer : public Bindable{
    public:
        VertexBuffer(GraphicsEngine& gfx, const void* vertexBuffers[], const size_t vertexSizes[], const bool cpuAccessFlags[], const int numVertexBuffers, const size_t numVertices);
        void bind(const GraphicsEngine& gfx) override;
        void update(const GraphicsEngine& gfx, int vertexBufferIndex, void* newData, size_t size);
    protected:
        size_t numVertices;
        std::vector<Microsoft::WRL::ComPtr<ID3D11Buffer>> vertexBufferPs;

        std::vector<UINT> strides;
        std::vector<UINT> offsets;
        std::vector<ID3D11Buffer*> preparedBuffers;
};