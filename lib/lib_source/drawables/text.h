#pragma once

#include "../drawable.h"
#include "../drawable_manager.h"
#include "../bindables/bindables_includes.h"

struct LIBRARY_API TextInitializerDesc : public DrawableInitializerDesc{
    float left_down_x;
    float left_down_y;
    float character_width;
    float character_height;
    std::string text;
    TextInitializerDesc(float left_down_x, float left_down_y, float character_width, float character_height, std::string text) noexcept;
};

class LIBRARY_API Text : public Drawable{
    public:
        Text(DrawableInitializerDesc& desc, GraphicsEngine& gfx);
        void initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const override;
        DirectX::XMMATRIX getTransformXM() const noexcept override;

        float left_down_x;
        float left_down_y;
        float character_width;
        float character_height;
        std::string text;
};

template class LIBRARY_API DrawableManager<Text>;