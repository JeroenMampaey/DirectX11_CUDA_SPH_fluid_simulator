#pragma once

#include "../bindable.h"
#include <string>

#ifndef DEFAULT_PIXEL_SHADERS_DIRECTORY
#define DEFAULT_PIXEL_SHADERS_DIRECTORY L"pixelshaders/"
#endif

#define PIXEL_PATH_CONCATINATED(original) DEFAULT_PIXEL_SHADERS_DIRECTORY original

class PixelShader : public Bindable{
    public:
        PixelShader(GraphicsEngine& gfx, const std::wstring& path);
        void bind(const GraphicsEngine& gfx) override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11PixelShader> pPixelShader;
};