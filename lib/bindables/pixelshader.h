#pragma once

#include "../bindable.h"
#include <string>
#include <wrl/client.h>

class PixelShader : public Bindable{
    public:
        PixelShader(GraphicsEngine& gfx, const std::wstring& path);
        void Bind(GraphicsEngine& gfx) override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11PixelShader> pPixelShader;
};