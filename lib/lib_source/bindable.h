#pragma once

#include "exceptions.h"
#include <memory>
#include "graphics_engine.h"

class Bindable{
    public:
        virtual void bind(const GraphicsEngine& gfx) = 0;
        virtual ~Bindable() = default;
    protected:
        static ID3D11DeviceContext* getContext(const GraphicsEngine& gfx) noexcept;
        static ID3D11Device* getDevice(const GraphicsEngine& gfx) noexcept;
};