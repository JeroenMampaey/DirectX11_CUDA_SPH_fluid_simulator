#include "filled_circle_drawer.h"
#include "../helpers.h"
#include "../bindables/bindables_includes.h"

#define INDEXED_NUM_VERTICES 4

#define NON_INDEXED_NUM_VERTICES 6

FilledCircleDrawer::~FilledCircleDrawer() noexcept = default;

FilledCircleDrawer::FilledCircleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    Drawer(pDrawerHelper)
{
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
    addSharedBind(std::make_unique<ConstantVertexBuffer>(helper->getGraphicsEngine(), vertexBuffers, vertexSizes, numVertices, 2));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(helper->getGraphicsEngine(), VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(std::make_unique<PixelShader>(helper->getGraphicsEngine(), PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 2, 1,
        1, 2, 3
    };

    addSharedBind(std::make_unique<IndexBuffer>(helper->getGraphicsEngine(), indices));
    setIndexCount(indices.size());

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {red, green, blue};

    addSharedBind(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(0, helper->getGraphicsEngine(), cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(std::make_unique<InputLayout>(helper->getGraphicsEngine(), ied, pvsbc));

    addSharedBind(std::make_unique<Topology>(helper->getGraphicsEngine(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, helper->getGraphicsEngine());
}

void FilledCircleDrawer::drawFilledCircle(float x, float y, float radius) const{
    pVcbuf->update(helper->getGraphicsEngine(),
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(radius, radius, 1.0f) * DirectX::XMMatrixTranslation(x, y, 0.0f) * helper->getGraphicsEngine().getView() * helper->getGraphicsEngine().getProjection()
        )
    );

    pVcbuf->bind(helper->getGraphicsEngine());

    bindSharedBinds(typeid(FilledCircleDrawer));
    drawIndexed();
}

FilledCircleInstanceDrawer::FilledCircleInstanceDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    Drawer(pDrawerHelper)
{
    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(helper->getGraphicsEngine(), VERTEX_PATH_CONCATINATED(L"VertexShader3.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(std::make_unique<PixelShader>(helper->getGraphicsEngine(), PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {red, green, blue};

    addSharedBind(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(0, helper->getGraphicsEngine(), cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"InstancePos", 0, DXGI_FORMAT_R32G32B32_FLOAT, 2, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1}
    };
    addSharedBind(std::make_unique<InputLayout>(helper->getGraphicsEngine(), ied, pvsbc));

    addSharedBind(std::make_unique<Topology>(helper->getGraphicsEngine(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    setVertexCount(NON_INDEXED_NUM_VERTICES);

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, helper->getGraphicsEngine());
}

FilledCircleInstanceDrawer::~FilledCircleInstanceDrawer() noexcept = default;

/*std::shared_ptr<FilledCircleInstanceBuffer> FilledCircleInstanceDrawer::createCpuAccessibleBuffer(int numberOfCircles, float radius){
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
    std::shared_ptr<FilledCircleInstanceBuffer> pBuffer = std::shared_ptr<FilledCircleInstanceBuffer>(new FilledCircleInstanceBuffer(this, bufferUidCounter, std::make_unique<CpuMappableVertexBuffer>(helper->getGraphicsEngine(), vertexBuffers, vertexSizes, cpuAccessFlags, numVertices, 3), numberOfCircles, radius));
    buffersMap.insert({bufferUidCounter, pBuffer});
    bufferUidCounter++;
    return pBuffer;
}*/

void FilledCircleInstanceDrawer::drawFilledCircleBuffer(FilledCircleInstanceBuffer& buffer){
    if(&buffer.helper->getGraphicsEngine()!=&helper->getGraphicsEngine()){
        throw std::exception("Tried drawing a FilledCircleInstanceBuffer with a wrong GraphicsEngine.");
    }

    if(buffer.isMapped){
        throw std::exception("Tried drawing a FilledCircleInstanceBuffer that is still mapped.");
    }

    pVcbuf->update(helper->getGraphicsEngine(),
        DirectX::XMMatrixTranspose(
            helper->getGraphicsEngine().getView() * helper->getGraphicsEngine().getProjection()
        )
    );

    pVcbuf->bind(helper->getGraphicsEngine());

    buffer.pVBuf->bind(helper->getGraphicsEngine());

    setInstanceCount(buffer.numberOfCircles);
    
    bindSharedBinds(typeid(FilledCircleInstanceDrawer));
    drawInstanced();
}

FilledCircleInstanceBuffer::FilledCircleInstanceBuffer(std::shared_ptr<Helper> helper, std::unique_ptr<MappableVertexBuffer> pVBuf, int numberOfCircles, float radius) noexcept
    :
    GraphicsBoundObject(helper),
    pVBuf(std::move(pVBuf)),
    numberOfCircles(numberOfCircles),
    radius(radius)
{}

FilledCircleInstanceBuffer::~FilledCircleInstanceBuffer() noexcept = default;

DirectX::XMFLOAT3* FilledCircleInstanceBuffer::getMappedAccess(){
    if(!isMapped){
        mappedBuffer = static_cast<DirectX::XMFLOAT3*>(pVBuf->getMappedAccess(helper->getGraphicsEngine(), 2));
        isMapped = true;
    }

    return mappedBuffer;
}

void FilledCircleInstanceBuffer::unMap(){
    if(isMapped){
        pVBuf->unMap(helper->getGraphicsEngine(), 2);
        isMapped = false;
    }
}

CpuAccessibleFilledCircleInstanceBuffer::CpuAccessibleFilledCircleInstanceBuffer(std::shared_ptr<Helper> helper, int numberOfCircles, float radius)
    :
    FilledCircleInstanceBuffer(helper, initializationHelper(helper->getGraphicsEngine(), numberOfCircles, radius), numberOfCircles, radius)
{}

std::unique_ptr<MappableVertexBuffer> CpuAccessibleFilledCircleInstanceBuffer::initializationHelper(GraphicsEngine& gfx, int numberOfCircles, float radius){
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
    return std::make_unique<CpuMappableVertexBuffer>(gfx, vertexBuffers, vertexSizes, cpuAccessFlags, numVertices, 3);
}