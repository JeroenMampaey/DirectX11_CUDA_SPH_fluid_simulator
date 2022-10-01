#pragma once

#include "bindable.h"
#include <vector>
#include <memory>
#include "exports.h"

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