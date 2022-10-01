#pragma once

#include <vector>
#include <memory>
#include "exports.h"
#include "graphics_engine.h"

class Helper{
        friend class GraphicsEngine;
    public:
        virtual ~Helper() noexcept = default;

        Helper& operator=(const Helper& copy) = delete;
        Helper& operator=(Helper&& copy) = delete;

        GraphicsEngine& getGraphicsEngine();
    protected:
        Helper(GraphicsEngine* pGfx) noexcept;
        GraphicsEngine* pGfx = nullptr;
};

class DrawerHelper : public Helper{
        friend class GraphicsEngine;
    public:
        DrawerHelper& operator=(const DrawerHelper& copy) = delete;
        DrawerHelper& operator=(DrawerHelper&& copy) = delete;

        void drawIndexed(int numIndices) const;
        void drawInstanced(int numVertices, int numInstances) const;

        std::type_index getLastDrawer() const;
        void setLastDrawer(std::type_index newIndex) const;

    
    private:
        DrawerHelper(GraphicsEngine* pGfx) noexcept;
};

class BindableHelper : public Helper{
        friend class GraphicsEngine;
    public:
        BindableHelper& operator=(const BindableHelper& copy) = delete;
        BindableHelper& operator=(BindableHelper&& copy) = delete;

        ID3D11DeviceContext& getContext();
        ID3D11Device& getDevice();

    private:
        BindableHelper(GraphicsEngine* pGfx) noexcept;
};