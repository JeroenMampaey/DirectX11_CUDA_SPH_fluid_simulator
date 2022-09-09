#pragma once

#include <DirectXMath.h>
#include <memory>
#include "exports.h"

struct LIBRARY_API DrawableStateUpdateDesc{
    virtual ~DrawableStateUpdateDesc() = default;
};

struct LIBRARY_API DrawableStateInitializerDesc{
    virtual ~DrawableStateInitializerDesc() = default;
};

struct LIBRARY_API DrawableState{
    public:
        virtual DirectX::XMMATRIX getTransformXM() const noexcept = 0;
        //TODO: allow colors to depends on state of the Drawable
        //virtual std::shared_ptr<float[]> getColorInformation() const noexcept = 0;
        virtual void update(DrawableStateUpdateDesc& desc) noexcept = 0;
        virtual ~DrawableState() = default;
};