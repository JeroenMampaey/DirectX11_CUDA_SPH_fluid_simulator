#pragma once

#include "../drawable.h"
#include "../drawable_manager.h"
#include "../bindables/bindables_includes.h"

struct LIBRARY_API DXFilledRectangleInitializerDesc : public DrawableInitializerDesc{
    float x;
    float y;
    float width;
    float height;
    DXFilledRectangleInitializerDesc(float x, float y, float width, float height) noexcept;
};

class LIBRARY_API DXFilledRectangle : public Drawable{
    public:
        DXFilledRectangle(DrawableInitializerDesc& desc, GraphicsEngine& gfx);
        void initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const override;
        DirectX::XMMATRIX getModel() const noexcept override;

        float x;
        float y;
        float width;
        float height;
};

template class LIBRARY_API DrawableManager<DXFilledRectangle>;