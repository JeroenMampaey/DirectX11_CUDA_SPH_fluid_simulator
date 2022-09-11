#include "example_engine.h"

ExampleEngine::ExampleEngine(HWND hWnd, UINT syncInterval) : GraphicsEngine(hWnd, syncInterval)
{
    if(RATE_IS_INVALID(refreshRate)){
        throw std::exception("Refreshrate could not easily be found programmatically.");
    }
    dt = 1.0f/refreshRate;
    this->setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
    for(int i=0; i<20; i++){
        SquareStateInitializerDesc desc = {50.0f+50.0f*i, (float)(HEIGHT/2), 50.0f, 50.0f};
        squares.push_back(std::move(squareFactory.createDrawable(*this, desc)));
    }
    for(int i=0; i<10; i++){
        LineStateInitializerDesc desc = {i*100.0f, (i-2)*(i-2)*10.0f, (i+1)*100.0f, (i-1)*(i-1)*10.0f, 0.0f, 0.0f, 0.0f};
        lines.push_back(std::move(lineFactory.createDrawable(*this, desc)));
    }
}

void ExampleEngine::update(){
    square_velocity += 9.81f*dt;
    beginFrame(0.2f, 0.2f, 1.0f);
    for(auto& square : squares){
        SquareState& state = static_cast<SquareState&>(square->getState());
        SquareStateUpdateDesc desc = {state.x, state.y-square_velocity*dt};
        square->updateState(desc);
        square->draw(*this);
    }
    for(auto& line : lines){
        line->draw(*this);
    }
    endFrame();
}