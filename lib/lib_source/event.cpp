#include "event.h"

bool EventBus::subscribe(const EventType& descriptor, EventListener* listener) noexcept{
    return listeners[descriptor].insert(listener).second;
}

bool EventBus::unsubscribe(const EventType& descriptor, EventListener* listener) noexcept{
    auto it = listeners.find(descriptor);
    if(it != listeners.end()){
        auto it2 = it->second.find(listener);
        if(it2 != it->second.end()){
            it->second.erase(it2);
            return true;
        }
    }
    return false;
}

bool EventBus::unsubscribe(EventListener* listener) noexcept{
    bool retval = false;
    for(std::pair<const EventType,std::set<EventListener*>>& pair : listeners){
        retval = retval || unsubscribe(pair.first, listener);
    }
    return retval;
}

void EventBus::post(const Event& event) const{
    EventType type = event.type();

    if(listeners.find(type)==listeners.end()){
        return;
    }

    const std::set<EventListener*>& specialized_listeners = listeners.at(type);

    for(EventListener* listener : specialized_listeners){
        listener->handleEvent(event);
    }
}

void EventListener::subscribeTo(std::shared_ptr<EventBus> pEventBus, const EventType& descriptor) noexcept{
    bool wasNotAlreadySubscribed = pEventBus->subscribe(descriptor, this);
    if(wasNotAlreadySubscribed){
        subscribedEventBusses[pEventBus]++;
    }
}

void EventListener::unsubscribeFrom(std::shared_ptr<EventBus> pEventBus, const EventType& descriptor) noexcept{
    bool wasSubscribedToThis = pEventBus->unsubscribe(descriptor, this);
    if(wasSubscribedToThis){
        int counter = --subscribedEventBusses[pEventBus];
        if(counter==0){
            subscribedEventBusses.erase(pEventBus);
        }
    }
}

void EventListener::unsubscribeFrom(std::shared_ptr<EventBus> pEventBus) noexcept{
    bool wasSubscribedToThis = pEventBus->unsubscribe(this);
    if(wasSubscribedToThis){
        subscribedEventBusses.erase(pEventBus);
    }
}

EventListener::~EventListener() noexcept{
    for(std::pair<const std::shared_ptr<EventBus>, int>& pair : subscribedEventBusses){
        pair.first->unsubscribe(this);
    }
}