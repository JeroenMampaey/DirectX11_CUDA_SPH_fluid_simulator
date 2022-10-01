#pragma once

#include "bindable.h"

class Topology : public Bindable{
    public:
        Topology(std::shared_ptr<BindableHelper> helper, D3D11_PRIMITIVE_TOPOLOGY type);
        void bind() override;
    protected:
        D3D11_PRIMITIVE_TOPOLOGY type;
};