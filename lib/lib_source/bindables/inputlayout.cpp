#include "inputlayout.h"

InputLayout::InputLayout(GraphicsEngine& gfx, const std::vector<D3D11_INPUT_ELEMENT_DESC>& layout, ID3DBlob* pVertexShaderBytecode){
	HRESULT hr;
	GFX_THROW_FAILED(getDevice(gfx)->CreateInputLayout(layout.data(), (UINT)layout.size(), pVertexShaderBytecode->GetBufferPointer(), pVertexShaderBytecode->GetBufferSize(), &pInputLayout));
}

void InputLayout::bind(const GraphicsEngine& gfx){
	getContext(gfx)->IASetInputLayout(pInputLayout.Get());
}