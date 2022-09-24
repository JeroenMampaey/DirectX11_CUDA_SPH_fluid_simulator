#include "filled_circle_drawer.h"

#define NUM_VERTICES 4

FilledCircleDrawerInitializationArgs::FilledCircleDrawerInitializationArgs(float red, float green, float blue) noexcept
    :
    red(red),
    green(green),
    blue(blue)
{}

FilledCircleDrawer::FilledCircleDrawer(GraphicsEngine* pGfx, int uid, FilledCircleDrawerInitializationArgs args)
    :
    Drawer(pGfx, uid)
{
    const DirectX::XMFLOAT3 vertices[NUM_VERTICES] = {
        {-1.0f, -1.0f, 0.0f},
        {1.0f, -1.0f, 0.0f},
        {-1.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    };

    const DirectX::XMFLOAT2 texcoords[NUM_VERTICES] = {
        {-1.0f, -1.0f},
        {1.0f, -1.0f},
        {-1.0f, 1.0f},
        {1.0f, 1.0f}
    };

    const void* vertexBuffers[] = {(void*)vertices, (void*)texcoords};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3), sizeof(DirectX::XMFLOAT2)};
    const bool cpuAccessFlags[] = {false, false};
    addSharedBind(std::make_unique<VertexBuffer>(*pGfx, vertexBuffers, vertexSizes, cpuAccessFlags, 2, NUM_VERTICES));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(*pGfx, VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(std::make_unique<PixelShader>(*pGfx, PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 2, 1,
        1, 2, 3
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
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(std::make_unique<InputLayout>(*pGfx, ied, pvsbc));

    addSharedBind(std::make_unique<Topology>(*pGfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, *pGfx);
}

void FilledCircleDrawer::drawFilledCircle(float x, float y, float radius) const{
    if(pGfx==nullptr){
        throw std::exception("Tried drawing a FilledCircle with a GraphicsEngine that was already destroyed");
    }

    pVcbuf->update(*pGfx,
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(radius, radius, 1.0f) * DirectX::XMMatrixTranslation(x, y, 0.0f) * pGfx->getView() * pGfx->getProjection()
        )
    );

    pVcbuf->bind(*pGfx);

    draw();
}
