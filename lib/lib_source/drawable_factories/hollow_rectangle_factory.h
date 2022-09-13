#pragma once

#include "../drawable_factory.h"
#include "../bindables/bindables_includes.h"
#include "../exports.h"

struct LIBRARY_API HollowRectangleStateInitializerDesc : public DrawableStateInitializerDesc{
    float x;
    float y;
    float width;
    float height;
    float red;
    float green;
    float blue;
    HollowRectangleStateInitializerDesc(float x, float y, float width, float height, float red, float green, float blue) noexcept;
};

struct LIBRARY_API HollowRectangleState : public DrawableState{
    float x;
    float y;
    float width;
    float height;
    float red;
    float green;
    float blue;
    HollowRectangleState(float x, float y, float width, float height, float red, float green, float blue) noexcept;
    DirectX::XMMATRIX getTransformXM() const noexcept override;
};

class LIBRARY_API HollowRectangleFactory : public DrawableFactory<HollowRectangleFactory>{
    public:
        std::unique_ptr<DrawableState> getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept override;
        void initializeSharedBinds(GraphicsEngine& gfx) override;
        void initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const override;
};

template class LIBRARY_API DrawableFactory<HollowRectangleFactory>;