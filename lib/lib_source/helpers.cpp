#include "helpers.h"

GraphicsEngine& Helper::getGraphicsEngine(){
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a helper");
    }

    return *pGfx;
}

Helper::Helper(GraphicsEngine* pGfx) noexcept
    :
    pGfx(pGfx)
{}

void DrawerHelper::drawIndexed(int numIndices) const{
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a DrawerHelper");
    }

    pGfx->pContext->DrawIndexed(numIndices, 0, 0);   
}

void DrawerHelper::drawInstanced(int numVertices, int numInstances) const{
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a DrawerHelper");
    }

    pGfx->pContext->DrawInstanced(numVertices, numInstances, 0, 0);
}

std::type_index DrawerHelper::getLastDrawer() const{
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a DrawerHelper");
    }

    return pGfx->lastDrawer;
}

void DrawerHelper::setLastDrawer(std::type_index newIndex) const{
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a DrawerHelper");
    }

    pGfx->lastDrawer = newIndex;
}

DrawerHelper::DrawerHelper(GraphicsEngine* pGfx) noexcept
    :
    Helper(pGfx)
{}

ID3D11DeviceContext& BindableHelper::getContext(){
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a BindableHelper");
    }

    return *pGfx->pContext.Get();
}

ID3D11Device& BindableHelper::getDevice(){
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a BindableHelper");
    }

    return *pGfx->pDevice.Get();
}

BindableHelper::BindableHelper(GraphicsEngine* pGfx) noexcept
    :
    Helper(pGfx)
{}