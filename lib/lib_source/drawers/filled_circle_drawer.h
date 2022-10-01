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
        std::unordered_map<int, std::weak_ptr<FilledCircleInstanceBuffer>> buffersMap;
        int bufferUidCounter = 0;
};

class FilledCircleInstanceBuffer : public GraphicsBoundObject<Helper>{
        friend FilledCircleInstanceDrawer;
    public:
        LIBRARY_API virtual ~FilledCircleInstanceBuffer() noexcept;
        LIBRARY_API DirectX::XMFLOAT3* getMappedAccess();
        LIBRARY_API void unMap();

        const int numberOfCircles;
        float radius;

    private:
        std::unique_ptr<MappableVertexBuffer> pVBuf;
        bool isMapped = false;
        DirectX::XMFLOAT3* mappedBuffer = nullptr;

    protected:
        FilledCircleInstanceBuffer(std::shared_ptr<Helper> helper, std::unique_ptr<MappableVertexBuffer> pVBuf, int numberOfCircles, float radius) noexcept;
};

class CpuAccessibleFilledCircleInstanceBuffer : public FilledCircleInstanceBuffer{
    public:
#ifndef READ_FROM_LIB_HEADER
        CpuAccessibleFilledCircleInstanceBuffer(std::shared_ptr<Helper> helper, int numberOfCircles, float radius);
#endif

    private:
        static std::unique_ptr<MappableVertexBuffer> initializationHelper(GraphicsEngine& gfx, int numberOfCircles, float radius);
};