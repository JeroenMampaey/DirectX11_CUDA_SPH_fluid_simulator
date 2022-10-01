#pragma once

#include "bindable.h"

class Texture : public Bindable{
    public:
        struct Color{
            unsigned char b;
            unsigned char g;
            unsigned char r;
            unsigned char a;
            Color(unsigned char b, unsigned char g, unsigned char r, unsigned char a) noexcept; 
            Color() noexcept; 
        };
        Texture(std::shared_ptr<BindableHelper> helper, std::unique_ptr<Color[]> pBuffer, unsigned int width, unsigned int height);
        void bind() override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> pTextureView;
};