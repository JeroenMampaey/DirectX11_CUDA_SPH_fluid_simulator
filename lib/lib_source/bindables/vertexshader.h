#pragma once

#include <wrl/client.h>
#include <string>
#include "../bindable.h"

#ifndef DEFAULT_VERTEX_SHADERS_DIRECTORY
#define DEFAULT_VERTEX_SHADERS_DIRECTORY L"vertexshaders/"
#endif

#define VERTEX_PATH_CONCATINATED(original) DEFAULT_VERTEX_SHADERS_DIRECTORY original

class VertexShader : public Bindable
{
    public:
        VertexShader(GraphicsEngine& gfx,const std::wstring& path);
        void Bind(GraphicsEngine& gfx, DrawableState& drawableState) override;
        ID3DBlob* GetBytecode() const noexcept;
    protected:
        Microsoft::WRL::ComPtr<ID3DBlob> pBytecodeBlob;
        Microsoft::WRL::ComPtr<ID3D11VertexShader> pVertexShader;
};