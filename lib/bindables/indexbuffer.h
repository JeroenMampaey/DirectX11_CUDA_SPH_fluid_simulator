#pragma once

#include <wrl/client.h>
#include <vector>
#include "../bindable.h"

class IndexBuffer : public Bindable
{
    public:
        IndexBuffer(GraphicsEngine& gfx, const std::vector<unsigned short>& indices);
        void Bind(GraphicsEngine& gfx) override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11Buffer> pIndexBuffer;
};