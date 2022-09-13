#pragma once

#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include "exports.h"

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

class EventListener;

class LIBRARY_API EventBus{
    public:
        void subscribe(const EventType& descriptor, EventListener* listener) noexcept;
        void unsubscribe(const EventType& descriptor, EventListener* listener) noexcept;
        void unsubscribe(EventListener* listener) noexcept;
        void post(const Event& event) const noexcept;
    private:
        std::map<EventType, std::vector<EventListener*>> listeners;
};

class LIBRARY_API EventListener{
    public:
        EventListener(std::shared_ptr<EventBus> pEventBus) noexcept;
        virtual ~EventListener() noexcept;
        virtual void handleEvent(const Event& event) noexcept = 0;
    private:
        std::shared_ptr<EventBus> pEventBus;
};
