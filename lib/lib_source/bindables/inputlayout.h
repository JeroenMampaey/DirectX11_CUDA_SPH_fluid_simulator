#pragma once

#include "../bindable.h"
#include <wrl/client.h>
#include <vector>

class InputLayout : public Bindable
{
    public:
        InputLayout(GraphicsEngine& gfx, const std::vector<D3D11_INPUT_ELEMENT_DESC>& layout, ID3DBlob* pVertexShaderBytecode);
        void Bind(GraphicsEngine& gfx, DrawableState& DrawableState) override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11InputLayout> pInputLayout;
};