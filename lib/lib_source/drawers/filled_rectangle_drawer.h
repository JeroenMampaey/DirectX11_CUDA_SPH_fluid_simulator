#pragma once
#include "../exports.h"
#include <memory>
#include "../windows_includes.h"
#include <unordered_map>

class DrawerHelper;
template<class> class VertexConstantBuffer;

class FilledRectangleDrawer{
    public:
        LIBRARY_API void drawFilledRectangle(float x, float y, float width, float height) const;
        LIBRARY_API ~FilledRectangleDrawer() noexcept;

#ifndef READ_FROM_LIB_HEADER
        FilledRectangleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif
        
    private:
        std::shared_ptr<DrawerHelper> pDrawerHelper;
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};