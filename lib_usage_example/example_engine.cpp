#include "example_engine.h"

ExampleEngine::ExampleEngine(HWND hWnd, UINT msPerFrame) : GraphicsEngine(hWnd, msPerFrame)
{}

void ExampleEngine::update(){
    ClearBuffer(0.2, 0.2, 1.0);
    EndFrame();
}