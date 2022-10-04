#include "line_drawer.h"
#include "../helpers.h"
#include "../bindables/bindables_includes.h"

#define NUM_VERTICES 2

LineDrawer::~LineDrawer() noexcept = default;

LineDrawer::LineDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    Drawer(std::move(pDrawerHelper))
{
    GraphicsEngine& gfx = helper->getGraphicsEngine();
    
    const DirectX::XMFLOAT3 vertices[NUM_VERTICES] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    };

    const void* vertexBuffers[] = {(void*)vertices};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3)};
    const size_t numVertices[] = {NUM_VERTICES};
    addSharedBind(gfx.createNewGraphicsBoundObject<ConstantVertexBuffer>(vertexBuffers, vertexSizes, numVertices, 1));

    std::unique_ptr<VertexShader> pvs = gfx.createNewGraphicsBoundObject<VertexShader>(VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(gfx.createNewGraphicsBoundObject<PixelShader>(PIXEL_PATH_CONCATINATED(L"PixelShader1.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 1
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
    };
    addSharedBind(gfx.createNewGraphicsBoundObject<InputLayout>(ied, pvsbc));

    addSharedBind(gfx.createNewGraphicsBoundObject<Topology>(D3D11_PRIMITIVE_TOPOLOGY_LINELIST));

    pVcbuf = gfx.createNewGraphicsBoundObject<VertexConstantBuffer<DirectX::XMMATRIX>>(0);
}

void LineDrawer::drawLine(float x1, float y1, float x2, float y2) const{
    GraphicsEngine& gfx = helper->getGraphicsEngine();

    pVcbuf->update(
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(x2-x1, y2-y1, 1.0f) * DirectX::XMMatrixTranslation(x1, y1, 0.0f) * gfx.getView() * gfx.getProjection()
        )
    );

    pVcbuf->bind();

    bindSharedBinds();
    drawIndexed();
}

template LIBRARY_API std::unique_ptr<LineDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);