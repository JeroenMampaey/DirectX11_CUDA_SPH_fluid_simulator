#pragma once

#include "../bindables/bindables_includes.h"
#include "../drawer.h"

struct LIBRARY_API LineDrawerInitializationArgs{
    float red;
    float green;
    float blue;
    LineDrawerInitializationArgs(float red, float green, float blue) noexcept;
};

class LIBRARY_API LineDrawer : public Drawer{
        friend class GraphicsEngine;
    private:
        LineDrawer(GraphicsEngine* pGfx, int uid, LineDrawerInitializationArgs args);
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
    
    public:
        void drawLine(float x1, float y1, float x2, float y2) const;
};