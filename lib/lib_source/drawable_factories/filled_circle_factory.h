#pragma once

#include "../drawable_factory.h"
#include "../bindables/bindables_includes.h"
#include "../exports.h"

struct LIBRARY_API FilledCircleStateUpdateDesc : public DrawableStateUpdateDesc{
    float new_x;
    float new_y;
    FilledCircleStateUpdateDesc(float new_x, float new_y) noexcept;
};

struct LIBRARY_API FilledCircleStateInitializerDesc : public DrawableStateInitializerDesc{
    float x;
    float y;
    float radius;
    FilledCircleStateInitializerDesc(float x, float y, float radius) noexcept;
};

struct LIBRARY_API FilledCircleState : public DrawableState{
    float x;
    float y;
    float radius;
    FilledCircleState(float x, float y, float radius) noexcept;
    DirectX::XMMATRIX getTransformXM() const noexcept override;
    void update(DrawableStateUpdateDesc& desc) noexcept override;
};

class LIBRARY_API FilledCircleFactory : public DrawableFactory<FilledCircleFactory>{
    public:
        std::unique_ptr<DrawableState> getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept override;
        void initializeSharedBinds(GraphicsEngine& gfx) override;
        void initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const override;
};

template class LIBRARY_API DrawableFactory<FilledCircleFactory>;