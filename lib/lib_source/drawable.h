#pragma once

#include "bindable.h"
#include <vector>
#include <memory>
#include "exports.h"

class LIBRARY_API Drawable{
    public:
        virtual ~Drawable() = default;
        Drawable& operator=(const Drawable& copy) = delete;
        Drawable& operator=(Drawable&& copy) = delete;

        const std::vector<std::unique_ptr<Bindable>>& Drawable::getBinds() const noexcept;
        int Drawable::getIndexCount() const noexcept;

        virtual void initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const = 0;
        virtual DirectX::XMMATRIX getTransformXM() const noexcept = 0;
    
    private:
        std::vector<std::unique_ptr<Bindable>> binds;
        int indexCount = -1;
    
    protected:
        void addUniqueBind(std::unique_ptr<Bindable> bind) noexcept;
        void setIndexCount(int indexCount) noexcept;
};