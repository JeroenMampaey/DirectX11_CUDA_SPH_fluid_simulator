#pragma once

#include "drawer.h"

#define MAX_DYNAMIC_TEXT_DRAWER_STRLEN 256

class CpuMappableVertexBuffer;
class DrawerHelper;
template<class> class VertexConstantBuffer;

class DynamicTextDrawer : public Drawer{
    public:
        LIBRARY_API void drawDynamicText(const std::string& text, float left_down_x, float left_down_y, float char_width, float char_height);
        LIBRARY_API ~DynamicTextDrawer() noexcept;

#ifndef READ_FROM_LIB_HEADER
        DynamicTextDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif

    private:
        std::unique_ptr<CpuMappableVertexBuffer> pVBuf;
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};

class StaticScreenTextDrawer : public Drawer{
    public:
        LIBRARY_API void drawStaticScreenText() const;
        LIBRARY_API ~StaticScreenTextDrawer() noexcept;
    
#ifndef READ_FROM_LIB_HEADER
        StaticScreenTextDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, const std::string& text, float left_down_x, float left_down_y, float char_width, float char_height, float red, float green, float blue);
#endif
};