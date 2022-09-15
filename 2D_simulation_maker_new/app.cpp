#include "app.h"

App::App(Window& wnd) 
	:
    EventListener(wnd.getEventBus()),
	wnd(wnd)
{
    wnd.getGraphicsEngine().setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));

    wnd.getEventBus()->subscribe(EventType::MOUSE_MOVE_EVENT, this);
    wnd.getEventBus()->subscribe(EventType::MOUSE_LEFT_CLICK_EVENT, this);
    wnd.getEventBus()->subscribe(EventType::KEYBOARD_KEYDOWN_EVENT, this);

    keyCodeHelper['W'] = 2;
    keyCodeHelper['B'] = 3;
    keyCodeHelper['P'] = 4;
    keyCodeHelper['A'] = 9;
    keyCodeHelper[VK_LEFT] = 5;
    keyCodeHelper[VK_RIGHT] = 6;
    keyCodeHelper[VK_UP] = 7;
    keyCodeHelper[VK_DOWN] = 8;
}

void App::handleEvent(const Event& event){
    int automatonAction = -1;
    switch(event.type()){
        case EventType::MOUSE_MOVE_EVENT:
            {
                const MouseMoveEvent& castedEvent = static_cast<const MouseMoveEvent&>(event);
                mouseX = castedEvent.new_x;
                mouseY = castedEvent.new_y;
            }
            automatonAction = 1;
            break;
        case EventType::MOUSE_LEFT_CLICK_EVENT:
            {
                const MouseLeftClickEvent& castedEvent = static_cast<const MouseLeftClickEvent&>(event);
                mouseX = castedEvent.x;
                mouseY = castedEvent.y;
            }
            automatonAction = 0;
            break;
        case EventType::KEYBOARD_KEYDOWN_EVENT:
            {
                const KeyboardKeydownEvent& castedEvent = static_cast<const KeyboardKeydownEvent&>(event);
                if(keyCodeHelper.find(castedEvent.key)==keyCodeHelper.end()){
                    break;
                }
                automatonAction = keyCodeHelper[castedEvent.key];
            }
            break;
    }
    if(automatonAction==-1){
        return;
    }
    currentState = transitionTable[currentState][automatonAction];
    (this->*actionTable[currentState])();
}

void App::doNothing() {}

void App::addParticle() {
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            FilledCircleInitializerDesc desc = {(float)mouseX+2.5f*i*RADIUS, (float)mouseY+2.5f*j*RADIUS, RADIUS};
            particles.push_back(reinterpret_cast<FilledCircle*>(wnd.getGraphicsEngine().createDrawable(DrawableType::FILLED_CIRCLE, desc)));
        }
    }
}

void App::startLine(){
    LineInitializerDesc desc = {(float)mouseX, (float)mouseY, (float)mouseX, (float)mouseY, 0.0f, 0.0f, 0.0f};
    boundaries.push_back(reinterpret_cast<Line*>(wnd.getGraphicsEngine().createDrawable(DrawableType::LINE, desc)));
}

void App::moveLine(){
    Line* boundary = boundaries.back();
    boundary->x2 = mouseX;
    boundary->y2 = mouseY;
}

void App::startBox(){
    HollowRectangleInitializerDesc rectDesc = {(float)mouseX, (float)mouseY, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    pumps.push_back(reinterpret_cast<HollowRectangle*>(wnd.getGraphicsEngine().createDrawable(DrawableType::HOLLOW_RECTANGLE, rectDesc)));
    LineInitializerDesc lineDesc = {(float)mouseX, (float)mouseY, (float)mouseX+DEFAULT_PUMP_VELOCITY, (float)mouseY, 1.0f, 0.0f, 0.0f};
    pumpDirections.push_back(reinterpret_cast<Line*>(wnd.getGraphicsEngine().createDrawable(DrawableType::LINE, lineDesc)));
}

void App::moveBox(){
    HollowRectangle* pump = pumps.back();
    float old_x = pump->x-pump->width/2;
    float old_y = pump->y-pump->height/2;
    
    if(mouseX<old_x || mouseY<old_y){
        return;
    }

    pump->x = old_x + (mouseX-old_x)/2;
    pump->y = old_y + (mouseY-old_y)/2;
    pump->width = mouseX-old_x;
    pump->height = mouseY-old_y;

    Line* pumpDirection = pumpDirections.back();
    float change_x = old_x + (mouseX-old_x)/2 - pumpDirection->x1;
    float change_y = old_y + (mouseY-old_y)/2 - pumpDirection->y1;
    pumpDirection->x1 += change_x;
    pumpDirection->y1 += change_y;
    pumpDirection->x2 += change_x;
    pumpDirection->y2 += change_y;
}

void App::moveBoxDirectionLeft(){
    Line* pumpDirection = pumpDirections.back();
    float change_x = pumpDirection->x2-pumpDirection->x1;
    float change_y = pumpDirection->y2-pumpDirection->y1;
    pumpDirection->x2 = pumpDirection->x1+change_y;
    pumpDirection->y2 = pumpDirection->y1-change_x;
}

void App::moveBoxDirectionRight(){
    Line* pumpDirection = pumpDirections.back();
    float change_x = pumpDirection->x2-pumpDirection->x1;
    float change_y = pumpDirection->y2-pumpDirection->y1;
    pumpDirection->x2 = pumpDirection->x1-change_y;
    pumpDirection->y2 = pumpDirection->y1+change_x;
}

void App::moveBoxVelocityUp(){
    Line* pumpDirection = pumpDirections.back();
    float change_x = pumpDirection->x2-pumpDirection->x1;
    float change_y = pumpDirection->y2-pumpDirection->y1;
    pumpDirection->x2 = pumpDirection->x1+change_x*1.1f;
    pumpDirection->y2 = pumpDirection->y1+change_y*1.1f;
}

void App::moveBoxVelocityDown(){
    Line* pumpDirection = pumpDirections.back();
    float change_x = pumpDirection->x2-pumpDirection->x1;
    float change_y = pumpDirection->y2-pumpDirection->y1;
    pumpDirection->x2 = pumpDirection->x1+change_x/1.1f;
    pumpDirection->y2 = pumpDirection->y1+change_y/1.1f;
}

void App::store(){
    //TODO
}
