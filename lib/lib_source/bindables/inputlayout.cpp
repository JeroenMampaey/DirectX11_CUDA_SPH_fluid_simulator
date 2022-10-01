#include "inputlayout.h"

InputLayout::InputLayout(std::shared_ptr<BindableHelper> helper, const std::vector<D3D11_INPUT_ELEMENT_DESC>& layout, ID3DBlob* pVertexShaderBytecode)
	:
	Bindable(helper)
{
	HRESULT hr;
	GFX_THROW_FAILED(helper->getDevice().CreateInputLayout(layout.data(), (UINT)layout.size(), pVertexShaderBytecode->GetBufferPointer(), pVertexShaderBytecode->GetBufferSize(), &pInputLayout));
}

void InputLayout::bind(){
	helper->getContext().IASetInputLayout(pInputLayout.Get());
}