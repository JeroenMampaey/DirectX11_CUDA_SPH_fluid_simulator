#include "drawable_factory_base.h"
#include "drawable.h"

std::unique_ptr<Drawable> DrawableFactoryBase::callDrawableConstructor(GraphicsEngine& gfx, DrawableFactoryBase& creator, std::unique_ptr<DrawableState> pState){
    return std::unique_ptr<Drawable>(new Drawable(gfx, creator, std::move(pState)));
}

void DrawableFactoryBase::addUniqueBind(Drawable& drawable, std::unique_ptr<Bindable> bind) const noexcept{
    drawable.addUniqueBind(std::move(bind));
}

void DrawableFactoryBase::addSharedBind(Drawable& drawable, std::shared_ptr<Bindable> bind) const noexcept{
    drawable.addSharedBind(bind);
}

void DrawableFactoryBase::setIndexCount(Drawable& drawable, int indexCount) const noexcept{
    drawable.setIndexCount(indexCount);
}