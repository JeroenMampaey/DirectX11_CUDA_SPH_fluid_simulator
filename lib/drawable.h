#pragma once

#include "bindable.h"
#include <vector>
#include <memory>

template<class T>
class Drawable{
    public:
        Drawable(GraphicsEngine& gfx){}
        ~Drawable() noexcept{
            if (--refCount==0){
                sharedBinds.clear();
                sharedIndexCount = -1;
            }
        }
        virtual DirectX::XMMATRIX GetTransformXM() const noexcept = 0;
        void Draw(GraphicsEngine& gfx) const{
            if(indexCount==-1) return;

            for(auto& b : binds){
                b->Bind(gfx);
            }

            for(auto& b : sharedBinds){
                b->Bind(gfx);
            }

            gfx.DrawIndexed(indexCount);
        }
    protected:
        void initializeDrawable(GraphicsEngine& gfx){   //!TODO: fix this, temporary solution
            if(++refCount==1){
                createSharedBinds(gfx);
            }
            if(sharedIndexCount!=-1){
                indexCount = sharedIndexCount;
            }
        }
        virtual void createSharedBinds(GraphicsEngine& gfx) = 0;
        void AddBind(std::unique_ptr<Bindable> bind) noexcept{
            binds.push_back(std::move(bind));
        }
        void setIndexCount(int indexCount) noexcept{
            this->indexCount = indexCount;
        }
        static void AddSharedBind(std::unique_ptr<Bindable> sharedBind) noexcept{
            sharedBinds.push_back(std::move(sharedBind));
        }
        static void setSharedIndexCount(int sharedIndexCount) noexcept{
            Drawable<T>::sharedIndexCount = sharedIndexCount;
        }
    private:
        std::vector<std::unique_ptr<Bindable>> binds;
        int indexCount = -1;
        static std::vector<std::unique_ptr<Bindable>> sharedBinds;
        static int sharedIndexCount;
        static unsigned refCount;
};

template<class T> std::vector<std::unique_ptr<Bindable>> Drawable<T>::sharedBinds;
template<class T> int Drawable<T>::sharedIndexCount = -1;
template<class T> unsigned Drawable<T>::refCount = 0;