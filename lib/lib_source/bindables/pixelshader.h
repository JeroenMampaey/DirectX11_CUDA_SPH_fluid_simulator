#pragma once

#include "bindable.h"
#include <string>

#ifndef DEFAULT_PIXEL_SHADERS_DIRECTORY
#define DEFAULT_PIXEL_SHADERS_DIRECTORY L"pixelshaders/"
#endif

#define PIXEL_PATH_CONCATINATED(original) DEFAULT_PIXEL_SHADERS_DIRECTORY original

class PixelShader : public Bindable{
    public:
        PixelShader(std::shared_ptr<BindableHelper> pHelper, const std::wstring& path);
        void bind() const override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11PixelShader> pPixelShader;
};