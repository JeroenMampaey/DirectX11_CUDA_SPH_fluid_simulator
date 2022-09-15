#include "event.h"

void EventBus::subscribe(const EventType& descriptor, EventListener* listener) noexcept{
    listeners[descriptor].push_back(listener);
}

void EventBus::unsubscribe(const EventType& descriptor, EventListener* listener) noexcept{
    std::vector<EventListener*> vec = listeners[descriptor];
    vec.erase(std::remove(vec.begin(), vec.end(), listener), vec.end());
}

void EventBus::unsubscribe(EventListener* listener) noexcept{
    for(std::pair<const EventType,std::vector<EventListener*>>& pair : listeners){
        unsubscribe(pair.first, listener);
    }
}

void EventBus::post(const Event& event) const{
    EventType type = event.type();

    if(listeners.find(type)==listeners.end()){
        return;
    }

    const std::vector<EventListener*>& specialized_listeners = listeners.at(type);

    for(EventListener* listener : specialized_listeners){
        listener->handleEvent(event);
    }
}

EventListener::EventListener(std::shared_ptr<EventBus> pEventBus) noexcept
    :
    pEventBus(pEventBus)
{}

EventListener::~EventListener() noexcept{
    pEventBus->unsubscribe(this);
}