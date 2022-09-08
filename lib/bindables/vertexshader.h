#pragma once

#include <wrl/client.h>
#include <string>
#include "../bindable.h"

class VertexShader : public Bindable
{
    public:
        VertexShader(GraphicsEngine& gfx,const std::wstring& path);
        void Bind(GraphicsEngine& gfx) override;
        ID3DBlob* GetBytecode() const noexcept;
    protected:
        Microsoft::WRL::ComPtr<ID3DBlob> pBytecodeBlob;
        Microsoft::WRL::ComPtr<ID3D11VertexShader> pVertexShader;
};