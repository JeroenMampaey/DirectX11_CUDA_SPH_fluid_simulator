#pragma once

#include "../drawable.h"
#include "../drawable_manager.h"
#include "../bindables/bindables_includes.h"

struct LIBRARY_API FilledRectangleInitializerDesc : public DrawableInitializerDesc{
    float x;
    float y;
    float width;
    float height;
    FilledRectangleInitializerDesc(float x, float y, float width, float height) noexcept;
};

class LIBRARY_API FilledRectangle : public Drawable{
    public:
        FilledRectangle(DrawableInitializerDesc& desc, GraphicsEngine& gfx);
        void initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const override;
        DirectX::XMMATRIX getModel() const noexcept override;

        float x;
        float y;
        float width;
        float height;
};

template class LIBRARY_API DrawableManager<FilledRectangle>;