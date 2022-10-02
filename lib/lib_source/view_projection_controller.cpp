#include "view_projection_controller.h"
#include "helpers.h"

ViewProjectionController::~ViewProjectionController() noexcept = default;

void ViewProjectionController::setProjection(DirectX::FXMMATRIX& proj) const{
    helper->setProjection(proj);
}

void ViewProjectionController::setView(DirectX::FXMMATRIX& v) const{
    helper->setView(v);
}

DirectX::XMMATRIX ViewProjectionController::getProjection() const{
    return helper->getGraphicsEngine().getProjection();
}

DirectX::XMMATRIX ViewProjectionController::getView() const{
    return helper->getGraphicsEngine().getView();
}

ViewProjectionController::ViewProjectionController(std::shared_ptr<ViewProjectionControllerHelper> pViewProjectionControllerHelper) noexcept
    :
    GraphicsBoundObject(pViewProjectionControllerHelper)
{}

template LIBRARY_API std::unique_ptr<ViewProjectionController> GraphicsEngine::createNewGraphicsBoundObject();