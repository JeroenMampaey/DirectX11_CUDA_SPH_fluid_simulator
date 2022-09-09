#pragma once

#include "bindable.h"
#include <vector>
#include <memory>

class Drawable;

class LIBRARY_API DrawableFactoryBase{
    public:
        virtual void registerDrawable(GraphicsEngine& gfx) = 0;
        virtual void removeDrawable() noexcept = 0;

    protected:
        std::unique_ptr<Drawable> callDrawableConstructor(GraphicsEngine& gfx, DrawableFactoryBase& creator, std::unique_ptr<DrawableState> pState);
        void addUniqueBind(Drawable& drawable, std::unique_ptr<Bindable> bind) const noexcept;
        void addSharedBind(Drawable& drawable, std::shared_ptr<Bindable> bind) const noexcept;
        void setIndexCount(Drawable& drawable, int indexCount) const noexcept;
};