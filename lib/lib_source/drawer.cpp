#include "drawer.h"

Drawer::Drawer(GraphicsEngine* pGfx, int uid) noexcept : pGfx(pGfx), uid(uid)
{}

void Drawer::unbindGraphicsEngine() noexcept{
    pGfx= nullptr;
}

void Drawer::addSharedBind(std::unique_ptr<Bindable> bind) noexcept{
    sharedBinds.push_back(std::move(bind));
}

void Drawer::setIndexCount(int indexCount) noexcept{
    this->indexCount = indexCount;
}

void Drawer::draw() const{
    if(indexCount==-1){
        throw std::exception("Tried drawing an object with an invalid indexCount");
    }
    else if(pGfx==nullptr){
        throw std::exception("Tried drawing an object with a GraphicsEngine that was already destroyed");
    }
    
    if(pGfx->lastDrawer!=uid){
        for(auto& bind : sharedBinds){
            bind->bind(*pGfx);
        }
        pGfx->lastDrawer = uid;
    }

    pGfx->pContext->DrawIndexed(indexCount, 0, 0);
}

Drawer::~Drawer() noexcept{
    if(pGfx==nullptr){
        return;
    }

    pGfx->drawersMap.erase(uid);
}