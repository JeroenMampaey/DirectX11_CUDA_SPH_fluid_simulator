#pragma once

#include "../exceptions.h"
#include <memory>
#include "../graphics_engine.h"
#include "../helpers.h"

class Bindable : public GraphicsBoundObject<BindableHelper>{
    public:
        Bindable(std::shared_ptr<BindableHelper> helper) noexcept;
        virtual ~Bindable() = default;
        virtual void bind() = 0;
};