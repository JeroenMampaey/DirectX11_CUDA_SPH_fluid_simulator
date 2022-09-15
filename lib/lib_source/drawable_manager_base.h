#pragma once

#include "exports.h"

class Drawable;

struct LIBRARY_API DrawableInitializerDesc{ 
    virtual ~DrawableInitializerDesc() = default;
};

class LIBRARY_API DrawableManagerBase{
    public:
        virtual ~DrawableManagerBase() = default;
        virtual Drawable* createDrawable(DrawableInitializerDesc& desc) = 0;
        virtual int removeDrawable(Drawable* drawable) noexcept = 0;
        virtual void drawAll() const = 0;
};