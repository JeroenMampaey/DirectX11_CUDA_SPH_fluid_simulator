#pragma once

#include "drawable.h"

template<class T>
class LIBRARY_API DrawableFactory : public DrawableFactoryBase{
    public:
        std::unique_ptr<Drawable> createDrawable(GraphicsEngine& gfx, DrawableStateInitializerDesc& desc){
            std::shared_ptr<DrawableState> initialState = getInitialDrawableState(desc);
            std::unique_ptr<Drawable> newDrawable = callDrawableConstructor(gfx, *this, initialState);
            if(sharedIndexCount!=-1){
                setIndexCount(*newDrawable, sharedIndexCount);
            }
            for(std::shared_ptr<Bindable> bind : sharedBinds){
                DrawableFactoryBase::addSharedBind(*newDrawable, bind);
            }
            initializeBinds(gfx, *newDrawable);
            return newDrawable;
        }

        void registerDrawable(GraphicsEngine& gfx) override{
            if(++refCount==1){
                initializeSharedBinds(gfx);
            }
        }

        void removeDrawable() noexcept override{
            if(--refCount==0){
                sharedBinds.clear();
                sharedIndexCount = -1;
            }
        }

        virtual std::shared_ptr<DrawableState> getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept = 0;
        virtual void initializeSharedBinds(GraphicsEngine& gfx) = 0;
        virtual void initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const = 0;
        virtual ~DrawableFactory() = default;
    
    protected:
        void addSharedBind(std::shared_ptr<Bindable> newBind) noexcept{
            sharedBinds.push_back(newBind);
        }

        void setSharedIndexCount(int newCount) noexcept{
            sharedIndexCount = newCount;
        }
    
    private:
        unsigned refCount = 0;
        std::vector<std::shared_ptr<Bindable>> sharedBinds;
        int sharedIndexCount = -1;
};