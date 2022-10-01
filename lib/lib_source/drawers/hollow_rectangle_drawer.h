#pragma once
#include "drawer.h"

class DrawerHelper;
template<class> class VertexConstantBuffer;

class HollowRectangleDrawer : public Drawer{
    public:
        LIBRARY_API void drawHollowRectangle(float x, float y, float width, float height) const;
        LIBRARY_API ~HollowRectangleDrawer() noexcept;

#ifndef READ_FROM_LIB_HEADER
        HollowRectangleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif
        
    private:
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};