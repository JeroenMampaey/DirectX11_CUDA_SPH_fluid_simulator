#pragma once

#include <vector>
#include <memory>
#include "exports.h"
#include "graphics_engine.h"

class Helper{
        friend class GraphicsEngine;
    public:
        virtual ~Helper() noexcept = default;

        GraphicsEngine& getGraphicsEngine();

        // Warning: Nobody except a GraphicsEngine should be constructing an object of this type.
        // This constructor has only been made public to allow maximum efficiency of std::make_shared.
        Helper(GraphicsEngine* pGfx) noexcept;

    protected:
        GraphicsEngine* pGfx = nullptr;
};

class DrawerHelper : public Helper{
        friend class GraphicsEngine;
    public:
        void drawIndexed(int numIndices) const;
        void drawInstanced(int numVertices, int numInstances) const;

        int getLastDrawer() const;
        void setLastDrawer(int drawerUid) const;

        // Warning: Nobody except a GraphicsEngine should be constructing an object of this type.
        // This constructor has only been made public to allow maximum efficiency of std::make_shared.
        DrawerHelper(GraphicsEngine* pGfx) noexcept;
};

class BindableHelper : public Helper{
        friend class GraphicsEngine;
    public:
        ID3D11DeviceContext& getContext();
        ID3D11Device& getDevice();

        // Warning: Nobody except a GraphicsEngine should be constructing an object of this type.
        // This constructor has only been made public to allow maximum efficiency of std::make_shared.
        BindableHelper(GraphicsEngine* pGfx) noexcept;
};

class FrameControllerHelper : public Helper{
        friend class GraphicsEngine;
    public:
        void beginFrame(float red, float green, float blue) const;
        void drawFrame() const;

        // Warning: Nobody except a GraphicsEngine should be constructing an object of this type.
        // This constructor has only been made public to allow maximum efficiency of std::make_shared.
        FrameControllerHelper(GraphicsEngine* pGfx) noexcept;
};

class ViewProjectionControllerHelper : public Helper{
        friend class GraphicsEngine;
    public:
        void setProjection(DirectX::FXMMATRIX& proj) const;
        void setView(DirectX::FXMMATRIX& v) const;

        // Warning: Nobody except a GraphicsEngine should be constructing an object of this type.
        // This constructor has only been made public to allow maximum efficiency of std::make_shared.
        ViewProjectionControllerHelper(GraphicsEngine* pGfx) noexcept;
};