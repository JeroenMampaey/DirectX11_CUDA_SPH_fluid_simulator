#include "app.h"

App::App(Window& wnd) 
	:
	wnd(wnd)
{
    wnd.getGraphicsEngine().setProjection(DirectX::XMMatrixIdentity()* 
        DirectX::XMMatrixScaling(2.0f/((float)WIDTH), 2.0f/((float)HEIGHT), 1.0f)*
        DirectX::XMMatrixTranslation(-1.0f, -1.0f, 0.0f));

    subscribeTo(wnd.getEventBus(), EventType::MOUSE_MOVE_EVENT);
    subscribeTo(wnd.getEventBus(), EventType::MOUSE_LEFT_CLICK_EVENT);
    subscribeTo(wnd.getEventBus(), EventType::KEYBOARD_KEYDOWN_EVENT);

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
                mouseY = HEIGHT-castedEvent.new_y;
            }
            automatonAction = 1;
            break;
        case EventType::MOUSE_LEFT_CLICK_EVENT:
            {
                const MouseLeftClickEvent& castedEvent = static_cast<const MouseLeftClickEvent&>(event);
                mouseX = castedEvent.x;
                mouseY = HEIGHT-castedEvent.y;
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
    LineInitializerDesc boundaryDesc = {(float)mouseX, (float)mouseY, (float)mouseX, (float)mouseY, 0.0f, 0.0f, 0.0f};
    boundaries.push_back(reinterpret_cast<Line*>(wnd.getGraphicsEngine().createDrawable(DrawableType::LINE, boundaryDesc)));

    LineInitializerDesc normalDesc = {(float)mouseX, (float)mouseY, (float)mouseX, (float)mouseY, 1.0f, 0.0f, 0.0f};
    boundary_normals.push_back(reinterpret_cast<Line*>(wnd.getGraphicsEngine().createDrawable(DrawableType::LINE, normalDesc)));
}

void App::moveLine(){
    Line* boundary = boundaries.back();
    boundary->x2 = mouseX;
    boundary->y2 = mouseY;

    float length = sqrt((boundary->x2-boundary->x1)*(boundary->x2-boundary->x1)+(boundary->y2-boundary->y1)*(boundary->y2-boundary->y1));
    float nx = (boundary->y2-boundary->y1)/length;
    float ny = (boundary->x1-boundary->x2)/length;
    Line* boundary_normal = boundary_normals.back();
    boundary_normal->x1 = boundary->x1+(boundary->x2-boundary->x1)/2.0f;
    boundary_normal->y1 = boundary->y1+(boundary->y2-boundary->y1)/2.0f;
    boundary_normal->x2 = boundary_normal->x1+nx*30.0f;
    boundary_normal->y2 = boundary_normal->y1+ny*30.0f;
}

void App::startBox(){
    HollowRectangleInitializerDesc rectDesc = {(float)mouseX, (float)mouseY, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    pumps.push_back(reinterpret_cast<HollowRectangle*>(wnd.getGraphicsEngine().createDrawable(DrawableType::HOLLOW_RECTANGLE, rectDesc)));
    LineInitializerDesc lineDesc = {(float)mouseX, (float)mouseY, (float)mouseX+DEFAULT_PUMP_VELOCITY, (float)mouseY, 1.0f, 0.5f, 0.0f};
    pumpDirections.push_back(reinterpret_cast<Line*>(wnd.getGraphicsEngine().createDrawable(DrawableType::LINE, lineDesc)));
}

void App::moveBox(){
    HollowRectangle* pump = pumps.back();
    float old_x = pump->x-pump->width/2;
    float old_y = pump->y-pump->height/2;

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
    std::string file_content = "PARTICLES:\n";
    for(FilledCircle* particle : particles){
        file_content += std::to_string((int)particle->x) + " " + std::to_string((int)particle->y) + "\n";
    }
    file_content += "LINES:\n";
    for(Line* line : boundaries){
        if(line->x2!=line->x1 || line->y2!=line->y1){
            file_content += std::to_string((int)line->x1) + " " + std::to_string((int)line->y1) + " " + std::to_string((int)line->x2) + " " + std::to_string((int)line->y2) + "\n";
        }
    }
    file_content += "PUMPS:\n";
    for(int i=0; i<pumps.size(); i++){
        HollowRectangle* pump = pumps[i];
        if(pump->width>0.0f && pump->height>0.0f){
            Line* pumpDirection = pumpDirections[i];
            file_content += std::to_string((int)pump->x) + " " + std::to_string((int)pump->y) + " " + std::to_string((int)pump->width) + " " + std::to_string((int)pump->height) + " " + std::to_string((int)(pumpDirection->x2-pumpDirection->x1)) + " " + std::to_string((int)(pumpDirection->y2-pumpDirection->y1)) + "\n";
        }
    }

    // Since nothing is really going on on the screen, it does not really matter that this is the UI thread, 
    // else you would want to do this asynchronously
    HANDLE hFile;
    const char* dataBuffer = file_content.c_str();
    DWORD dwBytesToWrite = file_content.size();
    DWORD dwBytesWritten = 0;
    hFile = CreateFileA("../../simulation_layout/simulation2D.txt", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    { 
        throw std::exception("File handle for '../../simulation_layout/simulation2D.txt' was found invalid while attempting to store the simulation");
    }

    BOOL bErrorFlag = WriteFile(hFile, dataBuffer, dwBytesToWrite, &dwBytesWritten, NULL);

    if (FALSE == bErrorFlag)
    {
        throw std::exception("Writing to '../../simulation_layout/simulation2D.txt' failed while attempting to store the simulation");
    }

    if (dwBytesWritten != dwBytesToWrite)
    {
        throw std::exception("Not all data managed to be written to '../../simulation_layout/simulation2D.txt' while attempting to store the simulation");
    }

    CloseHandle(hFile);

    for(FilledCircle* particle : particles){
        wnd.getGraphicsEngine().removeDrawable(DrawableType::FILLED_CIRCLE, particle);
    }
    for(Line* boundary : boundaries){
        wnd.getGraphicsEngine().removeDrawable(DrawableType::LINE, boundary);
    }
    for(Line* boundary_normal : boundary_normals){
        wnd.getGraphicsEngine().removeDrawable(DrawableType::LINE, boundary_normal);
    }
    for(HollowRectangle* pump : pumps){
        wnd.getGraphicsEngine().removeDrawable(DrawableType::HOLLOW_RECTANGLE, pump);
    }
    for(Line* pumpDirection : pumpDirections){
        wnd.getGraphicsEngine().removeDrawable(DrawableType::LINE, pumpDirection);
    }

    particles.clear();
    boundaries.clear();
    boundary_normals.clear();
    pumps.clear();
    pumpDirections.clear();
}
