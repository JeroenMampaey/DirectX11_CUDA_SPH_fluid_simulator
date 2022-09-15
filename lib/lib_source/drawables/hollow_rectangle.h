#pragma once

#include "../drawable.h"
#include "../drawable_manager.h"
#include "../bindables/bindables_includes.h"

struct LIBRARY_API HollowRectangleInitializerDesc : public DrawableInitializerDesc{
    float x;
    float y;
    float width;
    float height;
    float red;
    float green;
    float blue;
    HollowRectangleInitializerDesc(float x, float y, float width, float height, float red, float green, float blue) noexcept;
};

class LIBRARY_API HollowRectangle : public Drawable{
    public:
        HollowRectangle(DrawableInitializerDesc& desc, GraphicsEngine& gfx);
        void initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const override;
        DirectX::XMMATRIX getTransformXM() const noexcept override;

        float x;
        float y;
        float width;
        float height;
        float red;
        float green;
        float blue;
};

template class LIBRARY_API DrawableManager<HollowRectangle>;