#include "example_engine.h"

ExampleEngine::ExampleEngine(HWND hWnd, UINT msPerFrame) : GraphicsEngine(hWnd, msPerFrame)
{
    this->SetProjection(DirectX::XMMatrixPerspectiveLH(1.0f, 0.546875f, 0.5f, 40.0f));
    mySquare = squareFactory.createDrawable(*this);
}

void ExampleEngine::update(){
    ClearBuffer(0.2f, 0.2f, 1.0f);
    mySquare->draw(*this);
    EndFrame();
}