#pragma once

#include "drawer.h"

class DrawerHelper;
template<class> class VertexConstantBuffer;

class LineDrawer : public Drawer{
    public:
        LIBRARY_API void drawLine(float x1, float y1, float x2, float y2) const;
        LIBRARY_API ~LineDrawer() noexcept;

#ifndef READ_FROM_LIB_HEADER
        LineDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif
    
    private:
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};