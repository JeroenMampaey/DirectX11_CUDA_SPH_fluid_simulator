#include "filled_circle_drawer.h"
#include "../drawer_helper.h"
#include "../bindables/bindables_includes.h"

#define INDEXED_NUM_VERTICES 4

#define NON_INDEXED_NUM_VERTICES 6

FilledCircleDrawer::~FilledCircleDrawer() noexcept = default;

FilledCircleDrawer::FilledCircleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    pDrawerHelper(pDrawerHelper)
{
    if(pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried making a FilledCircleDrawer with an invalid DrawerHelper.");
    }

    const DirectX::XMFLOAT3 vertices[] = {
        {-1.0f, -1.0f, 0.0f},
        {1.0f, -1.0f, 0.0f},
        {-1.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    };

    const DirectX::XMFLOAT2 texcoords[] = {
        {-1.0f, -1.0f},
        {1.0f, -1.0f},
        {-1.0f, 1.0f},
        {1.0f, 1.0f}
    };

    const void* vertexBuffers[] = {(void*)vertices, (void*)texcoords};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3), sizeof(DirectX::XMFLOAT2)};
    const size_t numVertices[] = {INDEXED_NUM_VERTICES, INDEXED_NUM_VERTICES};
    pDrawerHelper->addSharedBind(std::make_unique<ConstantVertexBuffer>(*pDrawerHelper->pGfx, vertexBuffers, vertexSizes, numVertices, 2));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(*pDrawerHelper->pGfx, VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    pDrawerHelper->addSharedBind(std::move(pvs));

    pDrawerHelper->addSharedBind(std::make_unique<PixelShader>(*pDrawerHelper->pGfx, PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 2, 1,
        1, 2, 3
    };

    pDrawerHelper->addSharedBind(std::make_unique<IndexBuffer>(*pDrawerHelper->pGfx, indices));
    pDrawerHelper->setIndexCount(indices.size());

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {red, green, blue};

    pDrawerHelper->addSharedBind(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(0, *pDrawerHelper->pGfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    pDrawerHelper->addSharedBind(std::make_unique<InputLayout>(*pDrawerHelper->pGfx, ied, pvsbc));

    pDrawerHelper->addSharedBind(std::make_unique<Topology>(*pDrawerHelper->pGfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, *pDrawerHelper->pGfx);
}

void FilledCircleDrawer::drawFilledCircle(float x, float y, float radius) const{
    if(pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried drawing a FilledCircle with an invalid DrawerHelper");
    }

    pVcbuf->update(*pDrawerHelper->pGfx,
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(radius, radius, 1.0f) * DirectX::XMMatrixTranslation(x, y, 0.0f) * pDrawerHelper->pGfx->getView() * pDrawerHelper->pGfx->getProjection()
        )
    );

    pVcbuf->bind(*pDrawerHelper->pGfx);

    pDrawerHelper->bindSharedBinds();
    pDrawerHelper->drawIndexed();
}

FilledCircleInstanceDrawer::FilledCircleInstanceDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    pDrawerHelper(pDrawerHelper)
{
    if(pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried making a FilledCircleInstanceDrawer with an invalid DrawerHelper.");
    }

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(*pDrawerHelper->pGfx, VERTEX_PATH_CONCATINATED(L"VertexShader3.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    pDrawerHelper->addSharedBind(std::move(pvs));

    pDrawerHelper->addSharedBind(std::make_unique<PixelShader>(*pDrawerHelper->pGfx, PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {red, green, blue};

    pDrawerHelper->addSharedBind(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(0, *pDrawerHelper->pGfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"InstancePos", 0, DXGI_FORMAT_R32G32B32_FLOAT, 2, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1}
    };
    pDrawerHelper->addSharedBind(std::make_unique<InputLayout>(*pDrawerHelper->pGfx, ied, pvsbc));

    pDrawerHelper->addSharedBind(std::make_unique<Topology>(*pDrawerHelper->pGfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    pDrawerHelper->setVertexCount(NON_INDEXED_NUM_VERTICES);

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, *pDrawerHelper->pGfx);
}

FilledCircleInstanceDrawer::~FilledCircleInstanceDrawer() noexcept{
    for(const std::pair<int, std::weak_ptr<FilledCircleInstanceBuffer>>& bufferPair : buffersMap){
        if(std::shared_ptr<FilledCircleInstanceBuffer> buffer = bufferPair.second.lock()){
            buffer->pDrawer = nullptr;
        }
    }
}

std::shared_ptr<FilledCircleInstanceBuffer> FilledCircleInstanceDrawer::createCpuAccessibleBuffer(int numberOfCircles, float radius){
    if(pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried making a CPU accessible FilledCircleInstanceBuffer with a FilledCircleInstanceDrawer that has an invalid DrawerHelper");
    }

    std::vector<DirectX::XMFLOAT3> vertices;
    std::vector<DirectX::XMFLOAT2> texcoords;
    std::vector<DirectX::XMFLOAT3> instancecoords;

    vertices.push_back({-radius, -radius, 0.0f});
    vertices.push_back({-radius, radius, 0.0f});
    vertices.push_back({radius, -radius, 0.0f});
    vertices.push_back({radius, -radius, 0.0f});
    vertices.push_back({-radius, radius, 0.0f});
    vertices.push_back({radius, radius, 0.0f});

    texcoords.push_back({-1.0f, -1.0f});
    texcoords.push_back({-1.0f, 1.0f});
    texcoords.push_back({1.0f, -1.0f});
    texcoords.push_back({1.0f, -1.0f});
    texcoords.push_back({-1.0f, 1.0f});
    texcoords.push_back({1.0f, 1.0f});

    for(int i=0; i<numberOfCircles; i++){
        instancecoords.push_back({0.0f, 0.0f, 0.0f});
    }

    const void* vertexBuffers[] = {(void*)vertices.data(), (void*)texcoords.data(), (void*)instancecoords.data()};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3), sizeof(DirectX::XMFLOAT2), sizeof(DirectX::XMFLOAT3)};
    const UINT cpuAccessFlags[] = {0, 0, D3D11_CPU_ACCESS_WRITE};
    const size_t numVertices[] = {NON_INDEXED_NUM_VERTICES, NON_INDEXED_NUM_VERTICES, static_cast<size_t>(numberOfCircles)};
    std::shared_ptr<FilledCircleInstanceBuffer> pBuffer = std::shared_ptr<FilledCircleInstanceBuffer>(new FilledCircleInstanceBuffer(this, bufferUidCounter, std::make_unique<CpuMappableVertexBuffer>(*pDrawerHelper->pGfx, vertexBuffers, vertexSizes, cpuAccessFlags, numVertices, 3), numberOfCircles, radius));
    buffersMap.insert({bufferUidCounter, pBuffer});
    bufferUidCounter++;
    return pBuffer;
}

void FilledCircleInstanceDrawer::drawFilledCircleBuffer(FilledCircleInstanceBuffer* buffer) const{
    if(buffer->pDrawer!=this){
        throw std::exception("DrawFilledCircleInstance was called with a buffer that was not made by the drawer itself.");
    }

    if(pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried drawing a FilledCircleInstanceBuffer with a GraphicsEngine that was already destroyed");
    }

    if(buffer->isMapped){
        throw std::exception("Tried drawing a FilledCircleInstanceBuffer that is still mapped.");
    }

    pVcbuf->update(*pDrawerHelper->pGfx,
        DirectX::XMMatrixTranspose(
            pDrawerHelper->pGfx->getView() * pDrawerHelper->pGfx->getProjection()
        )
    );

    pVcbuf->bind(*pDrawerHelper->pGfx);

    buffer->pVBuf->bind(*pDrawerHelper->pGfx);

    pDrawerHelper->setInstanceCount(buffer->numberOfCircles);
    
    pDrawerHelper->bindSharedBinds();
    pDrawerHelper->drawInstanced();
}

FilledCircleInstanceBuffer::FilledCircleInstanceBuffer(FilledCircleInstanceDrawer* pDrawer, int uid, std::unique_ptr<MappableVertexBuffer> pVBuf, int numberOfCircles, float radius) noexcept
    :
    pDrawer(pDrawer),
    uid(uid),
    pVBuf(std::move(pVBuf)),
    numberOfCircles(numberOfCircles),
    radius(radius)
{}

FilledCircleInstanceBuffer::~FilledCircleInstanceBuffer() noexcept{
    if(pDrawer==nullptr){
        return;
    }

    pDrawer->buffersMap.erase(uid);
}

DirectX::XMFLOAT3* FilledCircleInstanceBuffer::getMappedAccess(){
    if(pDrawer==nullptr){
        throw std::exception("Tried getting mapped access to a buffer from which the Drawer was already destroyed");
    }

    if(pDrawer->pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried getting mapped access to a buffer from which the Drawer has an invalid DrawerHelper");
    }

    if(!isMapped){
        mappedBuffer = static_cast<DirectX::XMFLOAT3*>(pVBuf->getMappedAccess(*pDrawer->pDrawerHelper->pGfx, 2));
        isMapped = true;
    }

    return mappedBuffer;
}

void FilledCircleInstanceBuffer::unMap(){
    if(pDrawer==nullptr){
        throw std::exception("Tried unmapping a buffer from which the Drawer was already destroyed");
    }

    if(pDrawer->pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried unmapping a buffer from which the Drawer has an invalid DrawerHelper");
    }

    if(isMapped){
        pVBuf->unMap(*pDrawer->pDrawerHelper->pGfx, 2);
        isMapped = false;
    }
}