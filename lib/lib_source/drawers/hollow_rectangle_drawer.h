#pragma once

#include "../bindables/bindables_includes.h"
#include "../drawer.h"

struct LIBRARY_API HollowRectangleDrawerInitializationArgs{
    float red;
    float green;
    float blue;
    HollowRectangleDrawerInitializationArgs(float red, float green, float blue) noexcept;
};

class LIBRARY_API HollowRectangleDrawer : public Drawer{
        friend class GraphicsEngine;
    private:
        HollowRectangleDrawer(GraphicsEngine* pGfx, int uid, HollowRectangleDrawerInitializationArgs args);
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
    
    public:
        void drawHollowRectangle(float x, float y, float width, float height) const;
};