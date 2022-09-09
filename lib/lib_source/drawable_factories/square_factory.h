#pragma once

#include "../drawable_factory.h"
#include "../bindables/bindables_includes.h"
#include "../exports.h"

struct LIBRARY_API SquareStateUpdateDesc : public DrawableStateUpdateDesc{
    float new_x;
    float new_y;
    SquareStateUpdateDesc(float new_x, float new_y) noexcept;
};

struct LIBRARY_API SquareStateInitializerDesc : public DrawableStateInitializerDesc{
    float x;
    float y;
    SquareStateInitializerDesc(float x, float y) noexcept;
};

struct LIBRARY_API SquareState : public DrawableState{
    float x;
    float y;
    SquareState(float x, float y) noexcept;
    DirectX::XMMATRIX getTransformXM() const noexcept override;
    void update(DrawableStateUpdateDesc& desc) noexcept override;
};

class LIBRARY_API SquareFactory : public DrawableFactory<SquareFactory>{
    public:
        std::shared_ptr<DrawableState> getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept override;
        void initializeSharedBinds(GraphicsEngine& gfx) override;
        void initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const override;
};

template class LIBRARY_API DrawableFactory<SquareFactory>;