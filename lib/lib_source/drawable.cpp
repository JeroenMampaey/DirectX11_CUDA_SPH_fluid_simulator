#include "drawable.h"

const std::vector<std::unique_ptr<Bindable>>& Drawable::getBinds() const noexcept{
    return binds;
}

const std::vector<std::shared_ptr<Bindable>>& Drawable::getSharedBinds() const noexcept{
    return sharedBinds;
}

const int Drawable::getIndexCount() const noexcept{
    return indexCount;
}

void Drawable::updateState(DrawableStateUpdateDesc& desc) noexcept{
    pState->update(desc);
}

DrawableState& Drawable::getState() const noexcept{
    return *pState;
}

Drawable::~Drawable() noexcept{
    creator.removeDrawable();
}

Drawable::Drawable(GraphicsEngine& gfx, DrawableFactoryBase& creator, std::unique_ptr<DrawableState> pState) 
    : 
    creator(creator),
    pState(std::move(pState))
{
    creator.registerDrawable(gfx);
}

void Drawable::addUniqueBind(std::unique_ptr<Bindable> bind) noexcept{
    binds.push_back(std::move(bind));
}

void Drawable::addSharedBind(std::shared_ptr<Bindable> bind) noexcept{
    sharedBinds.push_back(bind);
}

void Drawable::setIndexCount(int indexCount) noexcept{
    this->indexCount = indexCount;
}