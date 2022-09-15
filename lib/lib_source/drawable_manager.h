#pragma once

#include "drawable.h"
#include "graphics_engine.h"
#include <set>
#include <vector>
#include "drawable_manager_base.h"

template<class T>
struct pointer_comp {
  typedef std::true_type is_transparent;
  // helper does some magic in order to reduce the number of
  // pairs of types we need to know how to compare: it turns
  // everything into a pointer, and then uses `std::less<T*>`
  // to do the comparison:
  struct helper {
    T* ptr;
    helper():ptr(nullptr) {}
    helper(helper const&) = default;
    helper(T* p):ptr(p) {}
    template<class U, class...Ts>
    helper( std::unique_ptr<U, Ts...> const& up ):ptr(up.get()) {}
    // && optional: enforces rvalue use only
    bool operator<( helper o ) const {
      return std::less<T*>()( ptr, o.ptr );
    }
  };
  // without helper, we would need 2^n different overloads, where
  // n is the number of types we want to support (so, 8 with
  // raw pointers, unique pointers, and shared pointers).  That
  // seems silly:
  // && helps enforce rvalue use only
  bool operator()( helper const&& lhs, helper const&& rhs ) const {
    return lhs < rhs;
  }
};

template<typename T, typename std::enable_if<std::is_base_of<Drawable, T>::value>::type* = nullptr>
class LIBRARY_API DrawableManager : public DrawableManagerBase{
    private:
        GraphicsEngine& gfx;
        //std::set<std::unique_ptr<Drawable>> drawables;
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