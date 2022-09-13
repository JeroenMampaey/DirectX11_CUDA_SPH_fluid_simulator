#pragma once

#include "bindable.h"
#include "drawable_factory_base.h"
#include "state.h"
#include <vector>
#include <memory>
#include "exports.h"

class LIBRARY_API Drawable{
        friend DrawableFactoryBase;
    public:
        const std::vector<std::unique_ptr<Bindable>>& getBinds() const noexcept;
        const std::vector<std::shared_ptr<Bindable>>& getSharedBinds() const noexcept;
        const int getIndexCount() const noexcept;
        DrawableState& getState() const noexcept;
        ~Drawable() noexcept;
        Drawable& operator=(const Drawable& copy) = delete;
        Drawable& operator=(Drawable&& copy) = delete;
    
    private:
        Drawable(GraphicsEngine& gfx, DrawableFactoryBase& creator, std::unique_ptr<DrawableState> pState);
        void addUniqueBind(std::unique_ptr<Bindable> bind) noexcept;
        void addSharedBind(std::shared_ptr<Bindable> bind) noexcept;
        void setIndexCount(int indexCount) noexcept;

        DrawableFactoryBase& creator;
        std::vector<std::unique_ptr<Bindable>> binds;
        std::vector<std::shared_ptr<Bindable>> sharedBinds;
        std::unique_ptr<DrawableState> pState;
        int indexCount = -1;
};