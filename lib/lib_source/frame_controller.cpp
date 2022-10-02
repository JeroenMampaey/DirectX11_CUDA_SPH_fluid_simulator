#include "frame_controller.h"
#include "helpers.h"

FrameController::~FrameController() noexcept = default;

void FrameController::beginFrame() const{
    helper->beginFrame(red, green, blue);
}

void FrameController::drawFrame() const{
    helper->drawFrame();
}

FrameController::FrameController(std::shared_ptr<FrameControllerHelper> pFrameControllerHelper, float red, float green, float blue) noexcept
    :
    GraphicsBoundObject(std::move(pFrameControllerHelper)),
    red(red),
    green(green),
    blue(blue)
{}

template LIBRARY_API std::unique_ptr<FrameController> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);