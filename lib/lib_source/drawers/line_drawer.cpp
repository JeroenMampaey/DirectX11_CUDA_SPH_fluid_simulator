#include "line_drawer.h"
#include "../drawer_helper.h"
#include "../bindables/bindables_includes.h"

#define NUM_VERTICES 2

LineDrawer::~LineDrawer() noexcept = default;

LineDrawer::LineDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    pDrawerHelper(pDrawerHelper)
{
    if(pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried making a LineDrawer with an invalid DrawerHelper.");
    }

    const DirectX::XMFLOAT3 vertices[NUM_VERTICES] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    };

    const void* vertexBuffers[] = {(void*)vertices};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3)};
    const size_t numVertices[] = {NUM_VERTICES};
    pDrawerHelper->addSharedBind(std::make_unique<ConstantVertexBuffer>(*pDrawerHelper->pGfx, vertexBuffers, vertexSizes, numVertices, 1));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(*pDrawerHelper->pGfx, VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    pDrawerHelper->addSharedBind(std::move(pvs));

    pDrawerHelper->addSharedBind(std::make_unique<PixelShader>(*pDrawerHelper->pGfx, PIXEL_PATH_CONCATINATED(L"PixelShader1.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 1
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
    };
    pDrawerHelper->addSharedBind(std::make_unique<InputLayout>(*pDrawerHelper->pGfx, ied, pvsbc));

    pDrawerHelper->addSharedBind(std::make_unique<Topology>(*pDrawerHelper->pGfx, D3D11_PRIMITIVE_TOPOLOGY_LINELIST));

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, *pDrawerHelper->pGfx);
}

void LineDrawer::drawLine(float x1, float y1, float x2, float y2) const{
    if(pDrawerHelper->pGfx==nullptr){
        throw std::exception("Tried drawing a Line with an invalid DrawerHelper");
    }

    pVcbuf->update(*pDrawerHelper->pGfx,
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(x2-x1, y2-y1, 1.0f) * DirectX::XMMatrixTranslation(x1, y1, 0.0f) * pDrawerHelper->pGfx->getView() * pDrawerHelper->pGfx->getProjection()
        )
    );

    pVcbuf->bind(*pDrawerHelper->pGfx);

    pDrawerHelper->bindSharedBinds();
    pDrawerHelper->drawIndexed();
}