#include "filled_circle_drawer.h"
#include "../helpers.h"
#include "../bindables/bindables_includes.h"

#define INDEXED_NUM_VERTICES 4

#define NON_INDEXED_NUM_VERTICES 6

FilledCircleDrawer::~FilledCircleDrawer() noexcept = default;

FilledCircleDrawer::FilledCircleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    Drawer(std::move(pDrawerHelper))
{
    GraphicsEngine& gfx = helper->getGraphicsEngine();

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
    addSharedBind(gfx.createNewGraphicsBoundObject<ConstantVertexBuffer>(vertexBuffers, vertexSizes, numVertices, 2));

    std::unique_ptr<VertexShader> pvs = gfx.createNewGraphicsBoundObject<VertexShader>(VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(gfx.createNewGraphicsBoundObject<PixelShader>(PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 2, 1,
        1, 2, 3
    };

    addSharedBind(gfx.createNewGraphicsBoundObject<IndexBuffer>(indices));
    setIndexCount(indices.size());

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {red, green, blue};

    addSharedBind(gfx.createNewGraphicsBoundObject<PixelConstantBuffer<ConstantBuffer2>>(0, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(gfx.createNewGraphicsBoundObject<InputLayout>(ied, pvsbc));

    addSharedBind(gfx.createNewGraphicsBoundObject<Topology>(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    pVcbuf = gfx.createNewGraphicsBoundObject<VertexConstantBuffer<DirectX::XMMATRIX>>(0);
}

void FilledCircleDrawer::drawFilledCircle(float x, float y, float radius) const{
    GraphicsEngine& gfx = helper->getGraphicsEngine();

    pVcbuf->update(
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(radius, radius, 1.0f) * DirectX::XMMatrixTranslation(x, y, 0.0f) * gfx.getView() * gfx.getProjection()
        )
    );

    pVcbuf->bind();

    bindSharedBinds();
    drawIndexed();
}

FilledCircleInstanceDrawer::FilledCircleInstanceDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    Drawer(std::move(pDrawerHelper))
{
    GraphicsEngine& gfx = helper->getGraphicsEngine();

    std::unique_ptr<VertexShader> pvs = gfx.createNewGraphicsBoundObject<VertexShader>(VERTEX_PATH_CONCATINATED(L"VertexShader3.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(gfx.createNewGraphicsBoundObject<PixelShader>(PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {red, green, blue};

    addSharedBind(gfx.createNewGraphicsBoundObject<PixelConstantBuffer<ConstantBuffer2>>(0, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"InstancePos", 0, DXGI_FORMAT_R32G32B32_FLOAT, 2, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1}
    };
    addSharedBind(gfx.createNewGraphicsBoundObject<InputLayout>(ied, pvsbc));

    addSharedBind(gfx.createNewGraphicsBoundObject<Topology>(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    setVertexCount(NON_INDEXED_NUM_VERTICES);

    pVcbuf = gfx.createNewGraphicsBoundObject<VertexConstantBuffer<DirectX::XMMATRIX>>(0);
}

template<class T>
void FilledCircleInstanceDrawer::drawFilledCircleBuffer(FilledCircleInstanceBuffer<T>& buffer){
    GraphicsEngine& gfx = helper->getGraphicsEngine();

    if(&buffer.helper->getGraphicsEngine()!=&gfx){
        throw std::exception("Tried drawing a FilledCircleInstanceBuffer with a wrong GraphicsEngine.");
    }

    if(buffer.isMapped){
        throw std::exception("Tried drawing a FilledCircleInstanceBuffer that is still mapped.");
    }

    pVcbuf->update(
        DirectX::XMMatrixTranspose(
            gfx.getView() * gfx.getProjection()
        )
    );

    pVcbuf->bind();

    if(buffer.numberOfCircles>0){
        buffer.pVBuf->bind();
    }

    setInstanceCount(buffer.numberOfCircles);
    
    bindSharedBinds();
    drawInstanced();
}

FilledCircleInstanceDrawer::~FilledCircleInstanceDrawer() noexcept = default;

template<class T>
FilledCircleInstanceBuffer<T>::~FilledCircleInstanceBuffer() noexcept = default;

template<class T>
T* FilledCircleInstanceBuffer<T>::getMappedAccess(){
    if(numberOfCircles==0){
        throw std::exception("Tried getting mapped access to a FilledCircleInstanceBuffer that is empty");
    }

    if(!isMapped){
        mappedBuffer = static_cast<T*>(pVBuf->getMappedAccess(2));
        isMapped = true;
    }

    return mappedBuffer;
}

template<class T>
void FilledCircleInstanceBuffer<T>::unMap(){
    if(isMapped){
        pVBuf->unMap(2);
        isMapped = false;
    }
}

CpuAccessibleFilledCircleInstanceBuffer::CpuAccessibleFilledCircleInstanceBuffer(std::shared_ptr<Helper> pHelper, int numberOfCircles, float radius)
    :
    FilledCircleInstanceBuffer(std::move(pHelper), numberOfCircles, radius)
{
    if(numberOfCircles==0){
        return;
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
    pVBuf = helper->getGraphicsEngine().createNewGraphicsBoundObject<CpuMappableVertexBuffer>(vertexBuffers, vertexSizes, cpuAccessFlags, numVertices, 3);
}

CudaAccessibleFilledCircleInstanceBuffer::CudaAccessibleFilledCircleInstanceBuffer(std::shared_ptr<Helper> pHelper, int numberOfCircles, float radius)
    :
    FilledCircleInstanceBuffer(std::move(pHelper), numberOfCircles, radius)
{
    if(numberOfCircles==0){
        return;
    }
    
    std::vector<DirectX::XMFLOAT3> vertices;
    std::vector<DirectX::XMFLOAT2> texcoords;
    std::vector<DirectX::XMFLOAT4> instancecoords;

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
        instancecoords.push_back({0.0f, 0.0f, 0.0f, 0.0f});
    }

    const void* vertexBuffers[] = {(void*)vertices.data(), (void*)texcoords.data(), (void*)instancecoords.data()};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3), sizeof(DirectX::XMFLOAT2), sizeof(DirectX::XMFLOAT4)};
    const bool cudaAccessibilityMask[] = {false, false, true};
    const size_t numVertices[] = {NON_INDEXED_NUM_VERTICES, NON_INDEXED_NUM_VERTICES, static_cast<size_t>(numberOfCircles)};
    pVBuf = helper->getGraphicsEngine().createNewGraphicsBoundObject<CudaMappableVertexBuffer>(vertexBuffers, vertexSizes, cudaAccessibilityMask, numVertices, 3);
}

template LIBRARY_API std::unique_ptr<FilledCircleDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<FilledCircleInstanceDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<CpuAccessibleFilledCircleInstanceBuffer> GraphicsEngine::createNewGraphicsBoundObject(int numberOfCircles, float radius);
template LIBRARY_API std::unique_ptr<CudaAccessibleFilledCircleInstanceBuffer> GraphicsEngine::createNewGraphicsBoundObject(int numberOfCircles, float radius);

template class FilledCircleInstanceBuffer<DirectX::XMFLOAT3>;
template class FilledCircleInstanceBuffer<DirectX::XMFLOAT4>;

template LIBRARY_API void FilledCircleInstanceDrawer::drawFilledCircleBuffer(FilledCircleInstanceBuffer<DirectX::XMFLOAT3>& buffer);
template LIBRARY_API void FilledCircleInstanceDrawer::drawFilledCircleBuffer(FilledCircleInstanceBuffer<DirectX::XMFLOAT4>& buffer);