#include "app.h"

App::App(Window& wnd) 
	:
    EventListener(wnd.getEventBus()),
	wnd(wnd)
{
    if(RATE_IS_INVALID(wnd.getGraphicsEngine().getRefreshRate())){
        throw std::exception("Refreshrate could not easily be found programmatically.");
    }
    dt = 1.0f/wnd.getGraphicsEngine().getRefreshRate();
    wnd.getGraphicsEngine().setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));
    for(int i=0; i<20; i++){
        FilledRectangleStateInitializerDesc desc = {50.0f+50.0f*i, (float)(HEIGHT/2), 50.0f, 50.0f};
        filledRectangles.push_back(std::move(filledRectangleFactory.createDrawable(wnd.getGraphicsEngine(), desc)));
    }
    for(int i=0; i<10; i++){
        LineStateInitializerDesc desc = {i*100.0f, (i-2)*(i-2)*10.0f, (i+1)*100.0f, (i-1)*(i-1)*10.0f, 0.0f, 0.0f, 0.0f};
        lines.push_back(std::move(lineFactory.createDrawable(wnd.getGraphicsEngine(), desc)));
    }
    for(int i=0; i<20; i++){
        FilledCircleStateInitializerDesc desc = {50.0f+50.0f*i, (float)(HEIGHT/2)+50.0f, 25.0f};
        filledCircles.push_back(std::move(filledCircleFactory.createDrawable(wnd.getGraphicsEngine(), desc)));
    }
    for(int i=0; i<10; i++){
        HollowRectangleStateInitializerDesc desc = {(i+1)*100.0f, (i-1)*(i-1)*10.0f, 50.0f, 50.0f, 0.0f, 0.0f, 0.0f};
        hollowRectangles.push_back(std::move(hollowRectangleFactory.createDrawable(wnd.getGraphicsEngine(), desc)));
    }

    wnd.getEventBus()->subscribe(EventType::MOUSE_LEFT_CLICK_EVENT, this);
    wnd.getEventBus()->subscribe(EventType::MOUSE_RIGHT_CLICK_EVENT, this);
}

void App::showFrame(){
    velocity += 9.81f*dt;
    wnd.getGraphicsEngine().beginFrame(0.6f, 0.8f, 1.0f);
    for(auto& filledRectangle : filledRectangles){
        FilledRectangleState& state = static_cast<FilledRectangleState&>(filledRectangle->getState());
        FilledRectangleStateUpdateDesc desc = {state.x, state.y-velocity*dt};
        filledRectangle->updateState(desc);
        wnd.getGraphicsEngine().draw(*filledRectangle);
    }
    for(auto& line : lines){
        wnd.getGraphicsEngine().draw(*line);
    }
    for(auto& circle : filledCircles){
        wnd.getGraphicsEngine().draw(*circle);
    }
    for(auto& hollowRectange : hollowRectangles){
        wnd.getGraphicsEngine().draw(*hollowRectange);
    }
    wnd.getGraphicsEngine().endFrame();
}

void App::handleEvent(const Event& event) noexcept{
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
