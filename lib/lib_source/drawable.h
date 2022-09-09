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
        void draw(GraphicsEngine& gfx) const;
        void updateState(DrawableStateUpdateDesc& desc) noexcept;
        ~Drawable() noexcept;
    
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