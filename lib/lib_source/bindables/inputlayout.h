#pragma once

#include "bindable.h"
#include <vector>

class InputLayout : public Bindable{
    public:
        InputLayout(std::shared_ptr<BindableHelper> pHelper, const std::vector<D3D11_INPUT_ELEMENT_DESC>& layout, ID3DBlob* pVertexShaderBytecode);
        void bind() const override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11InputLayout> pInputLayout;
};