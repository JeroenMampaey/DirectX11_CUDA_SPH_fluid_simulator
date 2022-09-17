#pragma once

#include "../drawable.h"
#include "../drawable_manager.h"
#include "../bindables/bindables_includes.h"

struct LIBRARY_API DXLineInitializerDesc : public DrawableInitializerDesc{
    float x1;
    float y1;
    float x2;
    float y2;
    float red;
    float green;
    float blue;
    DXLineInitializerDesc(float x1, float y1, float x2, float y2, float red, float green, float blue) noexcept;
};

class LIBRARY_API DXLine : public Drawable{
    public:
        DXLine(DrawableInitializerDesc& desc, GraphicsEngine& gfx);
        void initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const override;
        DirectX::XMMATRIX getModel() const noexcept override;

        float x1;
        float y1;
        float x2;
        float y2;
        float red;
        float green;
        float blue;
};

template class LIBRARY_API DrawableManager<DXLine>;