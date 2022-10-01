#pragma once

#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include "exports.h"
#include <set>

struct LIBRARY_API Event{
    virtual ~Event() = default;
    virtual int type() const noexcept = 0;
};

class EventBus{
        friend class EventListener;
    public:
        LIBRARY_API void post(const Event& event) const;
    private:
        bool subscribe(const int descriptor, EventListener* listener) noexcept;
        bool unsubscribe(const int descriptor, EventListener* listener) noexcept;
        bool unsubscribe(EventListener* listener) noexcept;
        std::map<int, std::set<EventListener*>> listeners;
};

class EventListener{
    public:
        LIBRARY_API void subscribeTo(std::shared_ptr<EventBus> pEventBus, const int descriptor) noexcept;
        LIBRARY_API void unsubscribeFrom(std::shared_ptr<EventBus> pEventBus, const int descriptor) noexcept;
        LIBRARY_API void unsubscribeFrom(std::shared_ptr<EventBus> pEventBus) noexcept;
        LIBRARY_API virtual ~EventListener() noexcept;
        LIBRARY_API virtual void handleEvent(const Event& event) = 0;
    private:
        std::map<std::shared_ptr<EventBus>, int> subscribedEventBusses;
};
