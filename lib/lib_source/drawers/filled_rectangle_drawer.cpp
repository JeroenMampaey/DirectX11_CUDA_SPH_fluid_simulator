#include "filled_rectangle_drawer.h"
#include "../helpers.h"
#include "../bindables/bindables_includes.h"

#define NUM_VERTICES 4

FilledRectangleDrawer::~FilledRectangleDrawer() noexcept = default;

FilledRectangleDrawer::FilledRectangleDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    Drawer(pDrawerHelper)
{
    const DirectX::XMFLOAT3 vertices[NUM_VERTICES] = {
        {-0.5f, -0.5f, 0.0f},
        {0.5f, -0.5f, 0.0f},
        {-0.5f, 0.5f, 0.0f},
        {0.5f, 0.5f, 0.0f}
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
    };
    addSharedBind(std::make_unique<InputLayout>(helper->getGraphicsEngine(), ied, pvsbc));

    addSharedBind(std::make_unique<Topology>(helper->getGraphicsEngine(), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, helper->getGraphicsEngine());
}

void FilledRectangleDrawer::drawFilledRectangle(float x, float y, float width, float height) const{
    pVcbuf->update(helper->getGraphicsEngine(),
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(width, height, 1.0f) * DirectX::XMMatrixTranslation(x, y, 0.0f) * helper->getGraphicsEngine().getView() * helper->getGraphicsEngine().getProjection()
        )
    );

    pVcbuf->bind(helper->getGraphicsEngine());

    bindSharedBinds(typeid(FilledRectangleDrawer));
    drawIndexed();
}