#include "line_drawer.h"
#include "../helpers.h"
#include "../bindables/bindables_includes.h"

#define NUM_VERTICES 2

LineDrawer::~LineDrawer() noexcept = default;

LineDrawer::LineDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    Drawer(pDrawerHelper)
{
    const DirectX::XMFLOAT3 vertices[NUM_VERTICES] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    };

    const void* vertexBuffers[] = {(void*)vertices};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3)};
    const size_t numVertices[] = {NUM_VERTICES};
    addSharedBind(std::make_unique<ConstantVertexBuffer>(helper->getGraphicsEngine(), vertexBuffers, vertexSizes, numVertices, 1));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(helper->getGraphicsEngine(), VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(std::make_unique<PixelShader>(helper->getGraphicsEngine(), PIXEL_PATH_CONCATINATED(L"PixelShader1.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 1
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
    };
    addSharedBind(std::make_unique<InputLayout>(helper->getGraphicsEngine(), ied, pvsbc));

    addSharedBind(std::make_unique<Topology>(helper->getGraphicsEngine(), D3D11_PRIMITIVE_TOPOLOGY_LINELIST));

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, helper->getGraphicsEngine());
}

void LineDrawer::drawLine(float x1, float y1, float x2, float y2) const{
    pVcbuf->update(helper->getGraphicsEngine(),
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(x2-x1, y2-y1, 1.0f) * DirectX::XMMatrixTranslation(x1, y1, 0.0f) * helper->getGraphicsEngine().getView() * helper->getGraphicsEngine().getProjection()
        )
    );

    pVcbuf->bind(helper->getGraphicsEngine());

    bindSharedBinds(typeid(LineDrawer));
    drawIndexed();
}