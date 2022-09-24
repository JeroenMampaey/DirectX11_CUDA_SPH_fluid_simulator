#pragma once

#include "../bindables/bindables_includes.h"
#include "../drawer.h"

#define MAX_DYNAMIC_TEXT_DRAWER_STRLEN 256

struct LIBRARY_API DynamicTextDrawerInitializationArgs{
    float red;
    float green;
    float blue;
    DynamicTextDrawerInitializationArgs(float red, float green, float blue) noexcept;
};

class LIBRARY_API DynamicTextDrawer : public Drawer{
        friend class GraphicsEngine;
    private:
        DynamicTextDrawer(GraphicsEngine* pGfx, int uid, DynamicTextDrawerInitializationArgs args);
        std::unique_ptr<VertexBuffer> pVBuf;
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;

    public:
        void drawDynamicText(const std::string& text, float left_down_x, float left_down_y, float char_width, float char_height);
};

struct LIBRARY_API StaticScreenTextDrawerInitializationArgs{
    std::string text; 
    float left_down_x; 
    float left_down_y; 
    float char_width; 
    float char_height;
    float red;
    float green;
    float blue;
    StaticScreenTextDrawerInitializationArgs(std::string text, float left_down_x, float left_down_y, float char_width, float char_height, float red, float green, float blue) noexcept;
};

class LIBRARY_API StaticScreenTextDrawer : public Drawer{
        friend class GraphicsEngine;
    private:
        StaticScreenTextDrawer(GraphicsEngine* pGfx, int uid, StaticScreenTextDrawerInitializationArgs args);

    public:
        void drawStaticScreenText() const;
};