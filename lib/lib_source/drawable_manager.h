#pragma once

#include "drawable.h"
#include "graphics_engine.h"
#include <set>
#include <vector>
#include <algorithm>
#include "drawable_manager_base.h"

template<typename T, typename std::enable_if<std::is_base_of<Drawable, T>::value>::type* = nullptr>
class LIBRARY_API DrawableManager : public DrawableManagerBase{
    private:
        GraphicsEngine& gfx;
        std::vector<std::unique_ptr<Drawable>> drawables;
        std::vector<std::unique_ptr<Bindable>> sharedBinds;
        int sharedIndexCount = -1;

    public:
        DrawableManager& operator=(const DrawableManager& copy) = delete;
        DrawableManager& operator=(DrawableManager&& copy) = delete;
        DrawableManager(GraphicsEngine& gfx) : gfx(gfx) {};
        
        Drawable* createDrawable(DrawableInitializerDesc& desc) override{
            drawables.push_back(std::make_unique<T>(desc, gfx));
            auto final_val = --drawables.end();

            if(drawables.size()==1){
                (*final_val)->initializeSharedBinds(gfx, sharedBinds, sharedIndexCount);
            }

            return final_val->get();
        }

        int removeDrawable(Drawable* drawable) noexcept override{
            auto it = std::find_if(drawables.begin(), drawables.end(), [&](std::unique_ptr<Drawable>& p) { return p.get() == drawable;});
            if(it != drawables.end()){
                drawables.erase(it);
                if(drawables.size()==0){
                    sharedBinds.clear();
                }
                return 1;
            }
            return 0;
        }

        void drawAll() const override{
            for(auto& bindable : sharedBinds){
                bindable->bind(gfx);
            }
            for(auto& drawable : drawables){
                int indexCount = (sharedIndexCount==-1) ? drawable->getIndexCount() : sharedIndexCount;
                if(indexCount==-1){
                    continue;
                }
                for(auto& bindable : drawable->getBinds()){
                    bindable->bind(gfx);
                }
                gfx.drawIndexed(indexCount);
            }
        }

        ~DrawableManager() noexcept{
            sharedBinds.clear();
        }
};