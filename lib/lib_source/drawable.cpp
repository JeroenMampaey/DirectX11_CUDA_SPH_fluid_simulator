#include "drawable.h"

void Drawable::draw(GraphicsEngine& gfx) const{
    if(indexCount==-1) return;

    for(auto& b : binds){
        b->Bind(gfx, *pState);
    }

    for(std::shared_ptr<Bindable> b : sharedBinds){
        b->Bind(gfx, *pState);
    }

    gfx.DrawIndexed(indexCount);
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