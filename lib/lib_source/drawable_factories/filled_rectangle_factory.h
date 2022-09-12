#pragma once

#include "../drawable_factory.h"
#include "../bindables/bindables_includes.h"
#include "../exports.h"

struct LIBRARY_API FilledRectangleStateUpdateDesc : public DrawableStateUpdateDesc{
    float new_x;
    float new_y;
    FilledRectangleStateUpdateDesc(float new_x, float new_y) noexcept;
};

struct LIBRARY_API FilledRectangleStateInitializerDesc : public DrawableStateInitializerDesc{
    float x;
    float y;
    float width;
    float height;
    FilledRectangleStateInitializerDesc(float x, float y, float width, float height) noexcept;
};

struct LIBRARY_API FilledRectangleState : public DrawableState{
    float x;
    float y;
    float width;
    float height;
    FilledRectangleState(float x, float y, float width, float height) noexcept;
    DirectX::XMMATRIX getTransformXM() const noexcept override;
    void update(DrawableStateUpdateDesc& desc) noexcept override;
};

class LIBRARY_API FilledRectangleFactory : public DrawableFactory<FilledRectangleFactory>{
    public:
        std::unique_ptr<DrawableState> getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept override;
        void initializeSharedBinds(GraphicsEngine& gfx) override;
        void initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const override;
};

template class LIBRARY_API DrawableFactory<FilledRectangleFactory>;