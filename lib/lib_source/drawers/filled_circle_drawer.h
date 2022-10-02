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

class FilledCircleInstanceBuffer;

class FilledCircleInstanceDrawer : public Drawer{
    public:
        LIBRARY_API ~FilledCircleInstanceDrawer() noexcept;
        LIBRARY_API void drawFilledCircleBuffer(FilledCircleInstanceBuffer& buffer);

#ifndef READ_FROM_LIB_HEADER
        FilledCircleInstanceDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif

    private:
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};

//TODO
class FilledCircleInstanceBuffer : public GraphicsBoundObject<Helper>{
        friend FilledCircleInstanceDrawer;
    public:
        LIBRARY_API virtual ~FilledCircleInstanceBuffer() noexcept;
        //LIBRARY_API DirectX::XMFLOAT3* getMappedAccess();
        LIBRARY_API void* getMappedAccess();
        LIBRARY_API void unMap();

        const int numberOfCircles;
        const float radius;

    private:
        bool isMapped = false;
        //DirectX::XMFLOAT3* mappedBuffer = nullptr;
        void* mappedBuffer = nullptr;

    protected:
        FilledCircleInstanceBuffer(std::shared_ptr<Helper> pHelper, int numberOfCircles, float radius) noexcept;
        std::unique_ptr<MappableVertexBuffer> pVBuf;
};

class CpuAccessibleFilledCircleInstanceBuffer : public FilledCircleInstanceBuffer{
    public:
#ifndef READ_FROM_LIB_HEADER
        CpuAccessibleFilledCircleInstanceBuffer(std::shared_ptr<Helper> pHelper, int numberOfCircles, float radius);
#endif
};

class CudaAccessibleFilledCircleInstanceBuffer : public FilledCircleInstanceBuffer{
    public:
#ifndef READ_FROM_LIB_HEADER
        CudaAccessibleFilledCircleInstanceBuffer(std::shared_ptr<Helper> pHelper, int numberOfCircles, float radius);
#endif
};