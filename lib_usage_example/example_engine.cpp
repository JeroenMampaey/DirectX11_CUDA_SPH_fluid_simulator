#include "example_engine.h"

ExampleEngine::ExampleEngine(HWND hWnd, std::shared_ptr<EventBus> pEventBus) : GraphicsEngine(hWnd, SYNCINTERVAL), EventListener(pEventBus)
{
    if(RATE_IS_INVALID(refreshRate)){
        throw std::exception("Refreshrate could not easily be found programmatically.");
    }
    dt = 1.0f/refreshRate;
    this->setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
    for(int i=0; i<20; i++){
        FilledRectangleStateInitializerDesc desc = {50.0f+50.0f*i, (float)(HEIGHT/2), 50.0f, 50.0f};
        filledRectangles.push_back(std::move(filledRectangleFactory.createDrawable(*this, desc)));
    }
    for(int i=0; i<10; i++){
        LineStateInitializerDesc desc = {i*100.0f, (i-2)*(i-2)*10.0f, (i+1)*100.0f, (i-1)*(i-1)*10.0f, 0.0f, 0.0f, 0.0f};
        lines.push_back(std::move(lineFactory.createDrawable(*this, desc)));
    }
    for(int i=0; i<20; i++){
        FilledCircleStateInitializerDesc desc = {50.0f+50.0f*i, (float)(HEIGHT/2)+50.0f, 25.0f};
        filledCircles.push_back(std::move(filledCircleFactory.createDrawable(*this, desc)));
    }
    for(int i=0; i<10; i++){
        HollowRectangleStateInitializerDesc desc = {(i+1)*100.0f, (i-1)*(i-1)*10.0f, 50.0f, 50.0f, 0.0f, 0.0f, 0.0f};
        hollowRectangles.push_back(std::move(hollowRectangleFactory.createDrawable(*this, desc)));
    }
    pEventBus->subscribe(EventType::MOUSE_LEFT_CLICK_EVENT, this);
    pEventBus->subscribe(EventType::MOUSE_RIGHT_CLICK_EVENT, this);
}

void ExampleEngine::update(){
    velocity += 9.81f*dt;
    beginFrame(0.6f, 0.8f, 1.0f);
    for(auto& filledRectangle : filledRectangles){
        FilledRectangleState& state = static_cast<FilledRectangleState&>(filledRectangle->getState());
        FilledRectangleStateUpdateDesc desc = {state.x, state.y-velocity*dt};
        filledRectangle->updateState(desc);
        filledRectangle->draw(*this);
    }
    for(auto& line : lines){
        line->draw(*this);
    }
    for(auto& circle : filledCircles){
        circle->draw(*this);
    }
    for(auto& hollowRectange : hollowRectangles){
        hollowRectange->draw(*this);
    }
    endFrame();
}

void ExampleEngine::handleEvent(const Event& event) noexcept{
    switch(event.type()){
        case EventType::MOUSE_LEFT_CLICK_EVENT:
            for(auto& circle : filledCircles){
                FilledCircleState& state = static_cast<FilledCircleState&>(circle->getState());
                FilledCircleStateUpdateDesc desc = {state.x, state.y+50.0f};
                circle->updateState(desc);
            }
            break;
        case EventType::MOUSE_RIGHT_CLICK_EVENT:
            for(auto& circle : filledCircles){
                FilledCircleState& state = static_cast<FilledCircleState&>(circle->getState());
                FilledCircleStateUpdateDesc desc = {state.x, state.y-50.0f};
                circle->updateState(desc);
            }
            break;
        default:
            break;
    }
}