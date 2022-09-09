#pragma once

#include <DirectXMath.h>
#include <memory>

class LIBRARY_API DrawableStateUpdateDesc{
    protected:
        DrawableStateUpdateDesc() noexcept {};
};

class LIBRARY_API DrawableState{
    public:
        virtual DirectX::XMMATRIX getTransformXM() const noexcept = 0;
        //TODO: allow colors to depends on state of the Drawable
        //virtual std::shared_ptr<float[]> getColorInformation() const noexcept = 0;
        virtual void update(DrawableStateUpdateDesc& desc) noexcept = 0;
};