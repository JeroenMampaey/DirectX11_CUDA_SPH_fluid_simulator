#include "inputlayout.h"

InputLayout::InputLayout(std::shared_ptr<BindableHelper> pHelper, const std::vector<D3D11_INPUT_ELEMENT_DESC>& layout, ID3DBlob* pVertexShaderBytecode)
	:
	Bindable(std::move(pHelper))
{
	HRESULT hr;
	GFX_THROW_FAILED(helper->getDevice().CreateInputLayout(layout.data(), (UINT)layout.size(), pVertexShaderBytecode->GetBufferPointer(), pVertexShaderBytecode->GetBufferSize(), &pInputLayout));
}

void InputLayout::bind() const{
	helper->getContext().IASetInputLayout(pInputLayout.Get());
}