#pragma once

#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include "exports.h"
#include <set>

enum EventType{
    MOUSE_MOVE_EVENT = 0,
    MOUSE_LEFT_CLICK_EVENT = 1,
    MOUSE_RIGHT_CLICK_EVENT = 2,
    KEYBOARD_KEYDOWN_EVENT = 3
};

struct LIBRARY_API Event{
    virtual ~Event() = default;
    virtual EventType type() const noexcept = 0;
};

class LIBRARY_API EventBus{
        friend class EventListener;
    public:
        void post(const Event& event) const;
    private:
        bool subscribe(const EventType& descriptor, EventListener* listener) noexcept;
        bool unsubscribe(const EventType& descriptor, EventListener* listener) noexcept;
        bool unsubscribe(EventListener* listener) noexcept;
        std::map<EventType, std::set<EventListener*>> listeners;
};

class LIBRARY_API EventListener{
    public:
        void subscribeTo(std::shared_ptr<EventBus> pEventBus, const EventType& descriptor) noexcept;
        void unsubscribeFrom(std::shared_ptr<EventBus> pEventBus, const EventType& descriptor) noexcept;
        void unsubscribeFrom(std::shared_ptr<EventBus> pEventBus) noexcept;
        virtual ~EventListener() noexcept;
        virtual void handleEvent(const Event& event) = 0;
    private:
        std::map<std::shared_ptr<EventBus>, int> subscribedEventBusses;
};
