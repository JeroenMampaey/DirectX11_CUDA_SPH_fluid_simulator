#pragma once
#include "drawer.h"

class DrawerHelper;
template<class> class VertexConstantBuffer;

class FilledRectangleDrawer : public Drawer{
    public:
        LIBRARY_API void drawFilledRectangle(float x, float y, float width, float height) const;
        LIBRARY_API ~FilledRectangleDrawer() noexcept;

#ifndef READ_FROM_LIB_HEADER
        FilledRectangleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif
        
    private:
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};