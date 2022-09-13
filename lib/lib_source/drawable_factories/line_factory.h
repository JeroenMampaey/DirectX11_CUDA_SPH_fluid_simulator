#pragma once

#include "../drawable_factory.h"
#include "../bindables/bindables_includes.h"
#include "../exports.h"
#include <math.h>

struct LIBRARY_API LineStateInitializerDesc : public DrawableStateInitializerDesc{
    float x1;
    float y1;
    float x2;
    float y2;
    float red;
    float green;
    float blue;
    LineStateInitializerDesc(float x1, float y1, float x2, float y2, float red, float green, float blue) noexcept;
};

struct LIBRARY_API LineState : public DrawableState{
    float x1;
    float y1;
    float x2;
    float y2;
    float red;
    float green;
    float blue;
    LineState(float x1, float y1, float x2, float y2, float red, float green, float blue) noexcept;
    DirectX::XMMATRIX getTransformXM() const noexcept override;
};

class LIBRARY_API LineFactory : public DrawableFactory<LineFactory>{
    public:
        std::unique_ptr<DrawableState> getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept override;
        void initializeSharedBinds(GraphicsEngine& gfx) override;
        void initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const override;
};

template class LIBRARY_API DrawableFactory<LineFactory>;
