#pragma once

#include "graphics_engine.h"
#include "state.h"

class Bindable{
    public:
        virtual void Bind(GraphicsEngine& gfx, DrawableState& drawableState) = 0;
    protected:
        static ID3D11DeviceContext* GetContext(GraphicsEngine& gfx) noexcept;
        static ID3D11Device* GetDevice(GraphicsEngine& gfx) noexcept;
};