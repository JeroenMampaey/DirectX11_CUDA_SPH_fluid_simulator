#pragma once

#include "bindable.h"

class Topology : public Bindable{
    public:
        Topology(std::shared_ptr<BindableHelper> pHelper, D3D11_PRIMITIVE_TOPOLOGY type);
        void bind() const override;
    protected:
        D3D11_PRIMITIVE_TOPOLOGY type;
};