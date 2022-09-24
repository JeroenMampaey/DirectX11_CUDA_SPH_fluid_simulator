#pragma once

#include "../bindables/bindables_includes.h"
#include "../drawer.h"

struct LIBRARY_API FilledCircleDrawerInitializationArgs{
    float red;
    float green;
    float blue;
    FilledCircleDrawerInitializationArgs(float red, float green, float blue) noexcept;
};

class LIBRARY_API FilledCircleDrawer : public Drawer{
        friend class GraphicsEngine;
    private:
        FilledCircleDrawer(GraphicsEngine* pGfx, int uid, FilledCircleDrawerInitializationArgs args);
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
    
    public:
        void drawFilledCircle(float x, float y, float radius) const;
};