#include "example_engine.h"

ExampleEngine::ExampleEngine(HWND hWnd, UINT msPerFrame) : GraphicsEngine(hWnd, msPerFrame)
{
    this->setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
    for(int i=0; i<20; i++){
        SquareStateInitializerDesc desc = {50.0f+50.0f*i, (float)(HEIGHT/2), 50.0f, 50.0f};
        squares.push_back(std::move(squareFactory.createDrawable(*this, desc)));
    }
}

void ExampleEngine::update(){
    float dt = ((float)MS_PER_FRAME)/1000.0f;
    square_velocity += 98.1f*dt;

    clearBuffer(0.2f, 0.2f, 1.0f);
    for(auto& square : squares){
        SquareState& state = static_cast<SquareState&>(square->getState());
        SquareStateUpdateDesc desc = {state.x, state.y-square_velocity*dt};
        square->updateState(desc);
        square->draw(*this);
    }
    endFrame();
}