#pragma once

#include "../bindable.h"

class Topology : public Bindable{
    public:
        Topology(GraphicsEngine& gfx, D3D11_PRIMITIVE_TOPOLOGY type);
        void bind(const GraphicsEngine& gfx) override;
    protected:
        D3D11_PRIMITIVE_TOPOLOGY type;
};