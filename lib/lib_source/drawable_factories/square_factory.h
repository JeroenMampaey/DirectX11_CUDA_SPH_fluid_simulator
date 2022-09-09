#pragma once

#include "../drawable_factory.h"
#include "../bindables/bindables_includes.h"
#include "../exports.h"

class LIBRARY_API SquareStateUpdateDesc : public DrawableStateUpdateDesc{
    public:
        float new_x;
        float new_y;
        SquareStateUpdateDesc(float new_x, float new_y) noexcept;
};

class LIBRARY_API SquareState : public DrawableState{
    public:
        float x;
        float y;
        SquareState(float x, float y) noexcept;
        DirectX::XMMATRIX getTransformXM() const noexcept;
        void update(DrawableStateUpdateDesc& desc) noexcept;
};

class LIBRARY_API SquareFactory : public DrawableFactory<SquareFactory>{
    public:
        std::unique_ptr<DrawableState> getInitialDrawableState() const noexcept;
        void initializeSharedBinds(GraphicsEngine& gfx);
        void initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const;
};

template class LIBRARY_API DrawableFactory<SquareFactory>;