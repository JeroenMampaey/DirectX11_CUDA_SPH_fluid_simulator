#include "app.h"

App::App(Window& wnd) 
	:
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
        FilledRectangleInitializerDesc desc = {50.0f+50.0f*i, (float)(HEIGHT/2), 50.0f, 50.0f};
        Drawable* filledRectangle = wnd.getGraphicsEngine().createDrawable(DrawableType::FILLED_RECTANGLE, desc);
        filledRectangles.push_back(reinterpret_cast<FilledRectangle*>(filledRectangle));
    }
    for(int i=0; i<10; i++){
        LineInitializerDesc desc = {i*100.0f, (i-2)*(i-2)*10.0f, (i+1)*100.0f, (i-1)*(i-1)*10.0f, 0.0f, 0.0f, 0.0f};
        wnd.getGraphicsEngine().createDrawable(DrawableType::LINE, desc);
    }
    for(int i=0; i<20; i++){
        FilledCircleInitializerDesc desc = {50.0f+50.0f*i, (float)(HEIGHT/2)+50.0f, 25.0f};
        Drawable* filledCircle = wnd.getGraphicsEngine().createDrawable(DrawableType::FILLED_CIRCLE, desc);
        filledCircles.push_back(reinterpret_cast<FilledCircle*>(filledCircle));
    }
    for(int i=0; i<10; i++){
        HollowRectangleInitializerDesc desc = {(i+1)*100.0f, (i-1)*(i-1)*10.0f, 50.0f, 50.0f, 0.0f, 0.0f, 0.0f};
        wnd.getGraphicsEngine().createDrawable(DrawableType::HOLLOW_RECTANGLE, desc);
    }

    TextInitializerDesc desc = {0.0f, 0.0f, 100.0f, 100.0f, "0123456789012"};
    wnd.getGraphicsEngine().createDrawable(DrawableType::TEXT, desc);

    subscribeTo(wnd.getEventBus(), EventType::MOUSE_LEFT_CLICK_EVENT);
    subscribeTo(wnd.getEventBus(), EventType::MOUSE_RIGHT_CLICK_EVENT);
    subscribeTo(wnd.getEventBus(), EventType::KEYBOARD_KEYDOWN_EVENT);
}

void App::handleEvent(const Event& event) noexcept{
    switch(event.type()){
        case EventType::MOUSE_LEFT_CLICK_EVENT:
            for(FilledCircle* circle : filledCircles){
                circle->y += 50.0f;
            }
            break;
        case EventType::MOUSE_RIGHT_CLICK_EVENT:
            for(FilledCircle* circle : filledCircles){
                circle->y -= 50.0f;
            }
            break;
        case EventType::KEYBOARD_KEYDOWN_EVENT:
            {
                const KeyboardKeydownEvent& castedEvent = static_cast<const KeyboardKeydownEvent&>(event);
                handleKeyEvent(castedEvent.key);
            }
            break;
    }
}

void App::periodicCallback() noexcept{
    velocity += 9.81f*dt;
    for(FilledRectangle* filledRectange : filledRectangles){
        filledRectange->y -= velocity*dt;
    }
}

void App::handleKeyEvent(LPARAM key) noexcept{
    switch(key){
        case 'A':
            if(filledCircles.size()>0){
                FilledCircle* circle =  filledCircles.back();
                if(wnd.getGraphicsEngine().removeDrawable(DrawableType::FILLED_CIRCLE, circle)){
                    filledCircles.pop_back();
                }
            }
            break;
        case VK_LEFT:
            cameraPositionX -= 30.0f;
            wnd.getGraphicsEngine().setView(DirectX::XMMatrixTranslation(-cameraPositionX, 0.0f, 0.0f));
            break;
        case VK_RIGHT:
            cameraPositionX += 30.0f;
            wnd.getGraphicsEngine().setView(DirectX::XMMatrixTranslation(-cameraPositionX, 0.0f, 0.0f));
            break;
    }
}
