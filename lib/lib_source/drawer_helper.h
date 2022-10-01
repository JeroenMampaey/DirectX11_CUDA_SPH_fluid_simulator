#pragma once

#include "bindable.h"
#include <vector>
#include <memory>
#include "exports.h"

class DrawerHelper{
        friend class GraphicsEngine;
    public:
        virtual ~DrawerHelper() noexcept;

        DrawerHelper& operator=(const DrawerHelper& copy) = delete;
        DrawerHelper& operator=(DrawerHelper&& copy) = delete;

        GraphicsEngine* pGfx = nullptr;

        void addSharedBind(std::unique_ptr<Bindable> bind) noexcept;
        void setIndexCount(int indexCount) noexcept;
        void setInstanceCount(int instanceCount) noexcept;
        void setVertexCount(int vertexCount) noexcept;
        void bindSharedBinds() const;
        void drawIndexed() const;
        void drawInstanced() const;
    
    private:
        DrawerHelper(GraphicsEngine* pGfx, int uid) noexcept;

        int uid;
        std::vector<std::unique_ptr<Bindable>> sharedBinds;
        int indexCount = -1;
        int vertexCount = -1;
        int instanceCount = -1;
};