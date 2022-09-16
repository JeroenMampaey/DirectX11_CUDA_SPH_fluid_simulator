#pragma once

#include "../drawable.h"
#include "../drawable_manager.h"
#include "../bindables/bindables_includes.h"

struct LIBRARY_API FilledCircleInitializerDesc : public DrawableInitializerDesc{
    float x;
    float y;
    float radius;
    FilledCircleInitializerDesc(float x, float y, float radius) noexcept;
};

class LIBRARY_API FilledCircle : public Drawable{
    public:
        FilledCircle(DrawableInitializerDesc& desc, GraphicsEngine& gfx);
        void initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const override;
        DirectX::XMMATRIX getModel() const noexcept override;

        float x;
        float y;
        float radius;
};

template class LIBRARY_API DrawableManager<FilledCircle>;