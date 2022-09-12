#pragma once

#include <wrl/client.h>
#include "../bindable.h"

class Texture : public Bindable{
    public:
        struct Color{
            unsigned int dword;
            Color(unsigned char x, unsigned char r, unsigned char g, unsigned char b) noexcept; 
            Color() noexcept; 
        };
        Texture(GraphicsEngine& gfx, std::unique_ptr<Color[]> pBuffer, unsigned int width, unsigned int height);
        void bind(GraphicsEngine& gfx, DrawableState& drawableState) override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> pTextureView;
};