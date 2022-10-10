#include "event_handler_system.h"
#include "render_system.h"

#pragma comment(lib,"xmllite.lib")
#pragma comment(lib,"Shlwapi.lib")
#include "xmllite.h"
#include "Shlwapi.h"
#include <tchar.h>

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

void EventHandlerSystem::addParticleZone() {
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            entityManager.getParticleZones().push_back(ParticleZone((float)mouseX, (float)mouseY));
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
    Microsoft::WRL::ComPtr<IStream> pOutFileStream;
    Microsoft::WRL::ComPtr<IXmlWriter> pWriter;

    if(FAILED(SHCreateStreamOnFileA(SLD_PATH_CONCATINATED("simulation2D.txt"), STGM_CREATE | STGM_WRITE, &pOutFileStream))){
        throw std::exception("Could not create a stream for the simulation2D.txt file");
    }
    
    if(FAILED(CreateXmlWriter(__uuidof(IXmlWriter), (void**) &pWriter, NULL))){
        throw std::exception("Could not create an XML writer for the simulation2D.txt file");
    }

    if(FAILED(pWriter->SetOutput(pOutFileStream.Get()))){
        throw std::exception("Could not bind the XML writer for the simulation2D.txt file");
    }

    if(FAILED(pWriter->SetProperty(XmlWriterProperty_Indent, TRUE))){
        throw std::exception("Could not set the indent property for writing the simulation2D.txt file");
    }

    if(FAILED(pWriter->WriteStartDocument(XmlStandalone_Omit)) || FAILED(pWriter->WriteStartElement(NULL, L"Layout", NULL)) || FAILED(pWriter->WriteAttributeString(NULL, L"particleZoneRadius", NULL, std::to_wstring(PARTICLE_ZONE_RADIUS).c_str()))){
        throw std::exception("Difficulty writing the simulation2D.txt file");
    }

    for(ParticleZone& particleZone : entityManager.getParticleZones()){
        if(FAILED(pWriter->WriteStartElement(NULL, L"ParticleZone", NULL))
            || FAILED(pWriter->WriteAttributeString(NULL, L"x", NULL, std::to_wstring((int)particleZone.x).c_str()))
            || FAILED(pWriter->WriteAttributeString(NULL, L"y", NULL, std::to_wstring((int)particleZone.y).c_str()))
            || FAILED(pWriter->WriteEndElement())
        ){
            throw std::exception("Difficulty writing the simulation2D.txt file");
        }
    }

    for(Boundary& boundary : entityManager.getBoundaries()){
        if(boundary.x2!=boundary.x1 || boundary.y2!=boundary.y1){
            if(FAILED(pWriter->WriteStartElement(NULL, L"Boundary", NULL))
                || FAILED(pWriter->WriteAttributeString(NULL, L"x1", NULL, std::to_wstring((int)boundary.x1).c_str()))
                || FAILED(pWriter->WriteAttributeString(NULL, L"y1", NULL, std::to_wstring((int)boundary.y1).c_str()))
                || FAILED(pWriter->WriteAttributeString(NULL, L"x2", NULL, std::to_wstring((int)boundary.x2).c_str()))
                || FAILED(pWriter->WriteAttributeString(NULL, L"y2", NULL, std::to_wstring((int)boundary.y2).c_str()))
                || FAILED(pWriter->WriteEndElement())
            ){
                throw std::exception("Difficulty writing the simulation2D.txt file");
            }
        }
    }

    for(Pump& pump : entityManager.getPumps()){
        if(pump.x1!=pump.x2 && pump.y1!=pump.y2){
            if(FAILED(pWriter->WriteStartElement(NULL, L"Pump", NULL))
                || FAILED(pWriter->WriteAttributeString(NULL, L"xLow", NULL, std::to_wstring((int)min(pump.x1, pump.x2)).c_str()))
                || FAILED(pWriter->WriteAttributeString(NULL, L"xHigh", NULL, std::to_wstring((int)max(pump.x1, pump.x2)).c_str()))
                || FAILED(pWriter->WriteAttributeString(NULL, L"yLow", NULL, std::to_wstring((int)min(pump.y1, pump.y2)).c_str()))
                || FAILED(pWriter->WriteAttributeString(NULL, L"yHigh", NULL, std::to_wstring((int)max(pump.y1, pump.y2)).c_str()))
                || FAILED(pWriter->WriteAttributeString(NULL, L"velX", NULL, std::to_wstring(pump.vel_x).c_str()))
                || FAILED(pWriter->WriteAttributeString(NULL, L"velY", NULL, std::to_wstring(pump.vel_y).c_str()))
                || FAILED(pWriter->WriteEndElement())
            ){
                throw std::exception("Difficulty writing the simulation2D.txt file");
            }
        }
    }

    if(FAILED(pWriter->WriteFullEndElement()) || FAILED(pWriter->WriteEndDocument())){
        throw std::exception("Difficulty writing the simulation2D.txt file");
    }

    if(FAILED(pWriter->Flush())){
        throw std::exception("Difficulty flushing the simulation2D.txt file");
    }

    entityManager.getParticleZones().clear();
    entityManager.getBoundaries().clear();
    entityManager.getPumps().clear();
}