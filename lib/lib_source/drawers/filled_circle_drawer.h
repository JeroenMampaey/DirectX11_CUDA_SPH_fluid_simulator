#pragma once
#include "drawer.h"

class DrawerHelper;
template<class> class VertexConstantBuffer;
class MappableVertexBuffer;

class FilledCircleDrawer : public Drawer{
    public:
        LIBRARY_API void drawFilledCircle(float x, float y, float radius) const;
        LIBRARY_API ~FilledCircleDrawer() noexcept;

#ifndef READ_FROM_LIB_HEADER
        FilledCircleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif

    private:
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};

template<class T>
class FilledCircleInstanceBuffer : public GraphicsBoundObject<Helper>{
        friend class FilledCircleInstanceDrawer;
    public:
        const int numberOfCircles;
        const float radius;

        LIBRARY_API virtual ~FilledCircleInstanceBuffer() noexcept;
        LIBRARY_API T* getMappedAccess();
        LIBRARY_API void unMap();

    protected:
        FilledCircleInstanceBuffer(std::shared_ptr<Helper> pHelper, int numberOfCircles, float radius) noexcept
            :
            GraphicsBoundObject(std::move(pHelper)),
            numberOfCircles(numberOfCircles),
            radius(radius)
        {}

        std::unique_ptr<MappableVertexBuffer> pVBuf;

    private:
        bool isMapped = false;
        T* mappedBuffer = nullptr;
};

class FilledCircleInstanceDrawer : public Drawer{
    public:
        LIBRARY_API ~FilledCircleInstanceDrawer() noexcept;
        template<class T>
        LIBRARY_API void drawFilledCircleBuffer(FilledCircleInstanceBuffer<T>& buffer);

#ifndef READ_FROM_LIB_HEADER
        FilledCircleInstanceDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif

    private:
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};

class CpuAccessibleFilledCircleInstanceBuffer : public FilledCircleInstanceBuffer<DirectX::XMFLOAT3>{
    public:
#ifndef READ_FROM_LIB_HEADER
        CpuAccessibleFilledCircleInstanceBuffer(std::shared_ptr<Helper> pHelper, int numberOfCircles, float radius);
#endif
};

class CudaAccessibleFilledCircleInstanceBuffer : public FilledCircleInstanceBuffer<DirectX::XMFLOAT4>{
    public:
        using FilledCircleInstanceBuffer<DirectX::XMFLOAT4>::getMappedAccess;
#ifndef READ_FROM_LIB_HEADER
        CudaAccessibleFilledCircleInstanceBuffer(std::shared_ptr<Helper> pHelper, int numberOfCircles, float radius);
#endif
};