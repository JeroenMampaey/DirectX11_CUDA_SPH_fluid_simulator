#pragma once

#include "../exports.h"
#include <memory>
#include "../windows_includes.h"
#include <unordered_map>
#include "../graphics_engine.h"

class DrawerHelper;
class Bindable;

class Drawer : public GraphicsBoundObject<DrawerHelper>{
    protected:
        Drawer(std::shared_ptr<DrawerHelper> helper) noexcept;
        virtual ~Drawer() noexcept;
        void addSharedBind(std::unique_ptr<Bindable> bind) noexcept;
        void setIndexCount(int indexCount) noexcept;
        void setInstanceCount(int instanceCount) noexcept;
        void setVertexCount(int vertexCount) noexcept;
        void bindSharedBinds(std::type_index) const;
        void drawIndexed() const;
        void drawInstanced() const; 

    private:
        std::vector<std::unique_ptr<Bindable>> sharedBinds;
        int indexCount = -1;
        int vertexCount = -1;
        int instanceCount = -1;
};