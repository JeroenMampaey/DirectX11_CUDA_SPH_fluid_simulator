#pragma once

#include "../drawable.h"
#include "../drawable_manager.h"
#include "../bindables/bindables_includes.h"

struct LIBRARY_API DXHollowRectangleInitializerDesc : public DrawableInitializerDesc{
    float x;
    float y;
    float width;
    float height;
    float red;
    float green;
    float blue;
    DXHollowRectangleInitializerDesc(float x, float y, float width, float height, float red, float green, float blue) noexcept;
};

class LIBRARY_API DXHollowRectangle : public Drawable{
    public:
        DXHollowRectangle(DrawableInitializerDesc& desc, GraphicsEngine& gfx);
        void initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const override;
        DirectX::XMMATRIX getModel() const noexcept override;

        float x;
        float y;
        float width;
        float height;
        float red;
        float green;
        float blue;
};

template class LIBRARY_API DrawableManager<DXHollowRectangle>;