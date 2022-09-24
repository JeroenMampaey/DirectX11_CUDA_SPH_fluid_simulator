#pragma once

#include "../bindables/bindables_includes.h"
#include "../drawer.h"

struct LIBRARY_API FilledRectangleDrawerInitializationArgs{
    float red;
    float green;
    float blue;
    FilledRectangleDrawerInitializationArgs(float red, float green, float blue) noexcept;
};

class LIBRARY_API FilledRectangleDrawer : public Drawer{
        friend class GraphicsEngine;
    private:
        FilledRectangleDrawer(GraphicsEngine* pGfx, int uid, FilledRectangleDrawerInitializationArgs args);
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
    
    public:
        void drawFilledRectangle(float x, float y, float width, float height) const;
};