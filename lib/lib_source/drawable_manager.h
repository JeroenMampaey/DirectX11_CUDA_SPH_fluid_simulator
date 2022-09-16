#pragma once

#include "drawable.h"
#include "graphics_engine.h"
#include <set>
#include <vector>
#include "drawable_manager_base.h"

// https://stackoverflow.com/questions/18939882/raw-pointer-lookup-for-sets-of-unique-ptrs
template<class T>
struct pointer_comp {
  typedef std::true_type is_transparent;
  struct helper {
    T* ptr;
    helper():ptr(nullptr) {}
    helper(helper const&) = default;
    helper(T* p):ptr(p) {}
    template<class U, class...Ts>
    helper( std::unique_ptr<U, Ts...> const& up ):ptr(up.get()) {}
    bool operator<( helper o ) const {
      return std::less<T*>()( ptr, o.ptr );
    }
  };
  bool operator()( helper const&& lhs, helper const&& rhs ) const {
    return lhs < rhs;
  }
};

template<typename T, typename std::enable_if<std::is_base_of<Drawable, T>::value>::type* = nullptr>
class LIBRARY_API DrawableManager : public DrawableManagerBase{
    private:
        GraphicsEngine& gfx;
        std::set<std::unique_ptr<Drawable>, pointer_comp<Drawable>> drawables;
        std::vector<std::unique_ptr<Bindable>> sharedBinds;
        int sharedIndexCount = -1;

    public:
        DrawableManager& operator=(const DrawableManager& copy) = delete;
        DrawableManager& operator=(DrawableManager&& copy) = delete;
        DrawableManager(GraphicsEngine& gfx) : gfx(gfx) {};
        
        Drawable* createDrawable(DrawableInitializerDesc& desc) override{
            auto retval = drawables.insert(std::make_unique<T>(desc, gfx));

            if(drawables.size()==1){
                (*retval.first)->initializeSharedBinds(gfx, sharedBinds, sharedIndexCount);
            }

            return retval.first->get();
        }

        int removeDrawable(Drawable* drawable) noexcept override{
            auto it = drawables.find(drawable);
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