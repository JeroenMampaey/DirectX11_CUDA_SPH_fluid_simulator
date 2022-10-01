#include "bindable.h"

Bindable::Bindable(std::shared_ptr<BindableHelper> helper) noexcept
    :
    GraphicsBoundObject(helper)
{}