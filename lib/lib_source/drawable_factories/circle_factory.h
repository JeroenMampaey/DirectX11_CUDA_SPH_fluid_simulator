#pragma once

#include "../drawable_factory.h"
#include "../bindables/bindables_includes.h"
#include "../exports.h"

struct LIBRARY_API CircleStateUpdateDesc : public DrawableStateUpdateDesc{
    float new_x;
    float new_y;
    CircleStateUpdateDesc(float new_x, float new_y) noexcept;
};

struct LIBRARY_API CircleStateInitializerDesc : public DrawableStateInitializerDesc{
    float x;
    float y;
    float radius;
    CircleStateInitializerDesc(float x, float y, float radius) noexcept;
};

struct LIBRARY_API CircleState : public DrawableState{
    float x;
    float y;
    float radius;
    CircleState(float x, float y, float radius) noexcept;
    DirectX::XMMATRIX getTransformXM() const noexcept override;
    void update(DrawableStateUpdateDesc& desc) noexcept override;
};

class LIBRARY_API CircleFactory : public DrawableFactory<CircleFactory>{
    public:
        std::unique_ptr<DrawableState> getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept override;
        void initializeSharedBinds(GraphicsEngine& gfx) override;
        void initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const override;
};

template class LIBRARY_API DrawableFactory<CircleFactory>;