#include "drawer_helper.h"

DrawerHelper::DrawerHelper(GraphicsEngine* pGfx, int uid) noexcept 
    : 
    pGfx(pGfx), 
    uid(uid)
{}

void DrawerHelper::addSharedBind(std::unique_ptr<Bindable> bind) noexcept{
    sharedBinds.push_back(std::move(bind));
}

void DrawerHelper::setIndexCount(int indexCount) noexcept{
    this->indexCount = indexCount;
}

void DrawerHelper::setInstanceCount(int instanceCount) noexcept{
    this->instanceCount = instanceCount;
}

void DrawerHelper::setVertexCount(int vertexCount) noexcept{
    this->vertexCount = vertexCount;
}

void DrawerHelper::bindSharedBinds() const{
    if(pGfx==nullptr){
        throw std::exception("Tried drawing an object with a GraphicsEngine that was already destroyed");
    }

    if(pGfx->lastDrawer!=uid){
        for(auto& bind : sharedBinds){
            bind->bind(*pGfx);
        }
        pGfx->lastDrawer = uid;
    }
}

void DrawerHelper::drawIndexed() const{
    if(indexCount==-1){
        throw std::exception("Tried drawing an object with an invalid indexCount");
    }
    else if(pGfx==nullptr){
        throw std::exception("Tried drawing an object with a GraphicsEngine that was already destroyed");
    }

    pGfx->pContext->DrawIndexed(indexCount, 0, 0);
}

void DrawerHelper::drawInstanced() const{
    if(vertexCount==-1){
        throw std::exception("Tried drawing an object with an invalid vertexCount");
    }
    if(instanceCount==-1){
        throw std::exception("Tried drawing an object with an invalid instanceCount");
    }
    else if(pGfx==nullptr){
        throw std::exception("Tried drawing an object with a GraphicsEngine that was already destroyed");
    }

    pGfx->pContext->DrawInstanced(vertexCount, instanceCount, 0, 0);
}

DrawerHelper::~DrawerHelper() noexcept{
    if(pGfx==nullptr){
        return;
    }

    pGfx->drawerHelpersMap.erase(uid);
}