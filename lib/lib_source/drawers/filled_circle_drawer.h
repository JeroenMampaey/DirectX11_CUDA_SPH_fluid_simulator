#pragma once
#include "../exports.h"
#include <memory>
#include "../windows_includes.h"
#include <unordered_map>

class DrawerHelper;
template<class> class VertexConstantBuffer;
class MappableVertexBuffer;

class FilledCircleDrawer{
    public:
        LIBRARY_API void drawFilledCircle(float x, float y, float radius) const;
        LIBRARY_API ~FilledCircleDrawer() noexcept;

#ifndef READ_FROM_LIB_HEADER
        FilledCircleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif

    private:

        std::shared_ptr<DrawerHelper> pDrawerHelper;
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
};

class FilledCircleInstanceDrawer{
        friend class FilledCircleInstanceBuffer;
    public:
        LIBRARY_API ~FilledCircleInstanceDrawer() noexcept;
        LIBRARY_API std::shared_ptr<FilledCircleInstanceBuffer> createCpuAccessibleBuffer(int numberOfCircles, float radius);
        LIBRARY_API void drawFilledCircleBuffer(FilledCircleInstanceBuffer* buffer) const;

#ifndef READ_FROM_LIB_HEADER
        FilledCircleInstanceDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue);
#endif

    private:
        std::shared_ptr<DrawerHelper> pDrawerHelper;
        std::unique_ptr<VertexConstantBuffer<DirectX::XMMATRIX>> pVcbuf;
        std::unordered_map<int, std::weak_ptr<FilledCircleInstanceBuffer>> buffersMap;
        int bufferUidCounter = 0;
};

class FilledCircleInstanceBuffer{
        friend FilledCircleInstanceDrawer;
    public:
        LIBRARY_API ~FilledCircleInstanceBuffer() noexcept;
        LIBRARY_API DirectX::XMFLOAT3* getMappedAccess();
        LIBRARY_API void unMap();

        const int numberOfCircles;
        float radius;

    private:
        FilledCircleInstanceBuffer(FilledCircleInstanceDrawer* pDrawer, int uid, std::unique_ptr<MappableVertexBuffer> pVBuf, int numberOfCircles, float radius) noexcept;

        int uid;
        FilledCircleInstanceDrawer* pDrawer;
        std::unique_ptr<MappableVertexBuffer> pVBuf;
        bool isMapped = false;
        DirectX::XMFLOAT3* mappedBuffer = nullptr;
};