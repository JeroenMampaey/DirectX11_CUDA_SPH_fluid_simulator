#include "event_handler_system.h"
#include "render_system.h"

EventHandlerSystem::EventHandlerSystem(std::shared_ptr<EventBus> windowEventBus, EntityManager& entityManager) noexcept
    :
    entityManager(entityManager)
{
    subscribeTo(windowEventBus, WindowEventType::MOUSE_MOVE_EVENT);
    subscribeTo(windowEventBus, WindowEventType::MOUSE_LEFT_CLICK_EVENT);
    subscribeTo(windowEventBus, WindowEventType::KEYBOARD_KEYDOWN_EVENT);

    keyCodeHelper['W'] = 2;
    keyCodeHelper['B'] = 3;
    keyCodeHelper['P'] = 4;
    keyCodeHelper['A'] = 9;
    keyCodeHelper[VK_LEFT] = 5;
    keyCodeHelper[VK_RIGHT] = 6;
    keyCodeHelper[VK_UP] = 7;
    keyCodeHelper[VK_DOWN] = 8;
}

void EventHandlerSystem::handleEvent(const Event& event){
    int automatonAction = -1;
    switch(event.type()){
        case WindowEventType::MOUSE_MOVE_EVENT:
            {
                const MouseMoveEvent& castedEvent = static_cast<const MouseMoveEvent&>(event);
                mouseX = castedEvent.new_x;
                mouseY = HEIGHT-castedEvent.new_y;
            }
            automatonAction = 1;
            break;
        case WindowEventType::MOUSE_LEFT_CLICK_EVENT:
            {
                const MouseLeftClickEvent& castedEvent = static_cast<const MouseLeftClickEvent&>(event);
                mouseX = castedEvent.x;
                mouseY = HEIGHT-castedEvent.y;
            }
            automatonAction = 0;
            break;
        case WindowEventType::KEYBOARD_KEYDOWN_EVENT:
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

void EventHandlerSystem::doNothing() {}

void EventHandlerSystem::addParticle() {
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            entityManager.getParticles().push_back(Particle((float)mouseX+2.5f*i*RADIUS, (float)mouseY+2.5f*j*RADIUS));
        }
    }
}

void EventHandlerSystem::startLine(){
    entityManager.getBoundaries().push_back(Boundary((float)mouseX, (float)mouseY, (float)mouseX, (float)mouseY));
}

void EventHandlerSystem::moveLine(){
    Boundary& boundary = entityManager.getBoundaries().back();
    boundary.x2 = mouseX;
    boundary.y2 = mouseY;
}

void EventHandlerSystem::startBox(){
    entityManager.getPumps().push_back(Pump((float)mouseX, (float)mouseY, (float)mouseX, (float)mouseY));
}

void EventHandlerSystem::moveBox(){
    Pump& pump = entityManager.getPumps().back();
    pump.x2 = mouseX;
    pump.y2 = mouseY;
}

void EventHandlerSystem::moveBoxDirectionLeft(){
    Pump& pump = entityManager.getPumps().back();
    float new_vel_x = -pump.vel_y;
    float new_vel_y = pump.vel_x;
    pump.vel_x = new_vel_x;
    pump.vel_y = new_vel_y;
}

void EventHandlerSystem::moveBoxDirectionRight(){
    Pump& pump = entityManager.getPumps().back();
    float new_vel_x = pump.vel_y;
    float new_vel_y = -pump.vel_x;
    pump.vel_x = new_vel_x;
    pump.vel_y = new_vel_y;
}

void EventHandlerSystem::moveBoxVelocityUp(){
    Pump& pump = entityManager.getPumps().back();
    pump.vel_x *= 1.1f;
    pump.vel_y *= 1.1f;
}

void EventHandlerSystem::moveBoxVelocityDown(){
    Pump& pump = entityManager.getPumps().back();
    pump.vel_x /= 1.1f;
    pump.vel_y /= 1.1f;
}

void EventHandlerSystem::store(){
    std::string file_content = "PARTICLES:\n";
    for(Particle& particle : entityManager.getParticles()){
        file_content += std::to_string((int)particle.x) + " " + std::to_string((int)particle.y) + "\n";
    }
    file_content += "LINES:\n";
    for(Boundary& boundary : entityManager.getBoundaries()){
        if(boundary.x2!=boundary.x1 || boundary.y2!=boundary.y1){
            file_content += std::to_string((int)boundary.x1) + " " + std::to_string((int)boundary.y1) + " " + std::to_string((int)boundary.x2) + " " + std::to_string((int)boundary.y2) + "\n";
        }
    }
    file_content += "PUMPS:\n";
    for(Pump& pump : entityManager.getPumps()){
        if(pump.x1!=pump.x2 && pump.y1!=pump.y2){
            int first_x = (int)min(pump.x1, pump.x2);
            int second_x = (int)max(pump.x1, pump.x2);
            int first_y = (int)min(pump.y1, pump.y2);
            int second_y = (int)max(pump.y1, pump.y2);
            file_content += std::to_string(first_x) + " " + std::to_string(second_x) + " " + std::to_string(first_y) + " " + std::to_string(second_y) + " " + std::to_string((int)pump.vel_x) + " " + std::to_string((int)pump.vel_y) + "\n";
        }
    }

    // Since nothing is really going on on the screen, it does not really matter that this is the UI thread, 
    // else you would want to do this asynchronously
    HANDLE hFile;
    const char* dataBuffer = file_content.c_str();
    DWORD dwBytesToWrite = file_content.size();
    DWORD dwBytesWritten = 0;
    hFile = CreateFileA(SLD_PATH_CONCATINATED("simulation2D.txt"), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    {
        throw std::exception("File handle for '" SLD_PATH_CONCATINATED("simulation2D.txt") "' was found invalid while attempting to store the simulation");
    }

    BOOL bErrorFlag = WriteFile(hFile, dataBuffer, dwBytesToWrite, &dwBytesWritten, NULL);

    if (FALSE == bErrorFlag)
    {
        throw std::exception("Writing to '" SLD_PATH_CONCATINATED("simulation2D.txt") "' failed while attempting to store the simulation");
    }

    if (dwBytesWritten != dwBytesToWrite)
    {
        throw std::exception("Not all data managed to be written to '" SLD_PATH_CONCATINATED("simulation2D.txt") "' while attempting to store the simulation");
    }

    CloseHandle(hFile);

    entityManager.getParticles().clear();
    entityManager.getBoundaries().clear();
    entityManager.getPumps().clear();
}