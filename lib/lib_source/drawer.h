#pragma once

#include "bindable.h"
#include <vector>
#include <memory>
#include "exports.h"

class LIBRARY_API Drawer{
    public:
        virtual ~Drawer() = default;
        void unbindGraphicsEngine() noexcept;

        Drawer& operator=(const Drawer& copy) = delete;
        Drawer& operator=(Drawer&& copy) = delete;
    
    protected:
        Drawer(GraphicsEngine* pGfx, int uid) noexcept;
        void addSharedBind(std::unique_ptr<Bindable> bind) noexcept;
        void setIndexCount(int indexCount) noexcept;
        void draw() const;
        GraphicsEngine* pGfx = nullptr;
    
    private:
        int uid;
        std::vector<std::unique_ptr<Bindable>> sharedBinds;
        int indexCount = -1;
};