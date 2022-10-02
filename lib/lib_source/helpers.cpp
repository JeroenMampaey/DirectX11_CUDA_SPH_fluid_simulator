#include "helpers.h"
#include "drawers/drawers_includes.h"
#include "exceptions.h"

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

void FrameControllerHelper::beginFrame(float red, float green, float blue) const{
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a FrameControllerHelper");
    }

    pGfx->pContext->OMSetRenderTargets(1, pGfx->pTarget.GetAddressOf(), pGfx->pDSV.Get());
    const float color[] = {red, green, blue, 1.0f};
    pGfx->pContext->ClearRenderTargetView(pGfx->pTarget.Get(), color);
    pGfx->pContext->ClearDepthStencilView(pGfx->pDSV.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    pGfx->lastDrawer = typeid(InvalidDrawer);
}

void FrameControllerHelper::drawFrame() const{
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a FrameControllerHelper");
    }

    pGfx->screenDrawer->drawStaticScreenText();
    HRESULT hr;
    if(FAILED(hr = pGfx->pSwap->Present(pGfx->syncInterval, 0))){
        if(hr == DXGI_ERROR_DEVICE_REMOVED){
            throw GFX_DEVICE_REMOVED_EXCEPT(pGfx->pDevice->GetDeviceRemovedReason());
        }
        else{
            GFX_THROW_FAILED(hr);
        }
    }
}

FrameControllerHelper::FrameControllerHelper(GraphicsEngine* pGfx) noexcept
    :
    Helper(pGfx)
{}

void ViewProjectionControllerHelper::setProjection(DirectX::FXMMATRIX& proj) const{
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a ViewProjectionControllerHelper");
    }

    pGfx->projection = proj;
}

void ViewProjectionControllerHelper::setView(DirectX::FXMMATRIX& v) const{
    if(pGfx==nullptr){
        throw std::exception("Tried accessing a GraphicsEngine that was already destroyed from a ViewProjectionControllerHelper");
    }

    pGfx->view = v;
}

ViewProjectionControllerHelper::ViewProjectionControllerHelper(GraphicsEngine* pGfx) noexcept
    :
    Helper(pGfx)
{}