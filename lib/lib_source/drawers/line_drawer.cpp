#include "line_drawer.h"

#define NUM_VERTICES 2

LineDrawerInitializationArgs::LineDrawerInitializationArgs(float red, float green, float blue) noexcept
    :
    red(red),
    green(green),
    blue(blue)
{}

LineDrawer::LineDrawer(GraphicsEngine* pGfx, int uid, LineDrawerInitializationArgs args)
    :
    Drawer(pGfx, uid)
{
    const DirectX::XMFLOAT3 vertices[NUM_VERTICES] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    };

    const void* vertexBuffers[] = {(void*)vertices};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3)};
    const bool cpuAccessFlags[] = {false};
    addSharedBind(std::make_unique<VertexBuffer>(*pGfx, vertexBuffers, vertexSizes, cpuAccessFlags, 1, NUM_VERTICES));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(*pGfx, VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(std::make_unique<PixelShader>(*pGfx, PIXEL_PATH_CONCATINATED(L"PixelShader1.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 1
    };

    addSharedBind(std::make_unique<IndexBuffer>(*pGfx, indices));
    setIndexCount(indices.size());

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {args.red, args.green, args.blue};

    addSharedBind(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(0, *pGfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(std::make_unique<InputLayout>(*pGfx, ied, pvsbc));

    addSharedBind(std::make_unique<Topology>(*pGfx, D3D11_PRIMITIVE_TOPOLOGY_LINELIST));

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, *pGfx);
}

void LineDrawer::drawLine(float x1, float y1, float x2, float y2) const{
    if(pGfx==nullptr){
        throw std::exception("Tried drawing a Line with a GraphicsEngine that was already destroyed");
    }

    pVcbuf->update(*pGfx,
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(x2-x1, y2-y1, 1.0f) * DirectX::XMMatrixTranslation(x1, y1, 0.0f) * pGfx->getView() * pGfx->getProjection()
        )
    );

    pVcbuf->bind(*pGfx);

    draw();
}