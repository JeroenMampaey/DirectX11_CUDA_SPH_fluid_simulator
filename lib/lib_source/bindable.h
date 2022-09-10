#pragma once

#include "graphics_engine.h"
#include "state.h"

class Bindable{
    public:
        virtual void bind(GraphicsEngine& gfx, DrawableState& drawableState) = 0;
        virtual ~Bindable() = default;
    protected:
        static ID3D11DeviceContext* getContext(GraphicsEngine& gfx) noexcept;
        static ID3D11Device* getDevice(GraphicsEngine& gfx) noexcept;
};