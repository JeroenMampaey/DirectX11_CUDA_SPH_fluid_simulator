#include "filled_rectangle_drawer.h"

#define NUM_VERTICES 4

FilledRectangleDrawerInitializationArgs::FilledRectangleDrawerInitializationArgs(float red, float green, float blue) noexcept
    :
    red(red),
    green(green),
    blue(blue)
{}

FilledRectangleDrawer::FilledRectangleDrawer(GraphicsEngine* pGfx, int uid, FilledRectangleDrawerInitializationArgs args)
    :
    Drawer(pGfx, uid)
{
    const DirectX::XMFLOAT3 vertices[NUM_VERTICES] = {
        {-0.5f, -0.5f, 0.0f},
        {0.5f, -0.5f, 0.0f},
        {-0.5f, 0.5f, 0.0f},
        {0.5f, 0.5f, 0.0f}
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
    };
    addSharedBind(std::make_unique<InputLayout>(*pGfx, ied, pvsbc));

    addSharedBind(std::make_unique<Topology>(*pGfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    pVcbuf = std::make_unique<VertexConstantBuffer<DirectX::XMMATRIX>>(0, *pGfx);
}

void FilledRectangleDrawer::drawFilledRectangle(float x, float y, float width, float height) const{
    if(pGfx==nullptr){
        throw std::exception("Tried drawing a FilledRectangle with a GraphicsEngine that was already destroyed");
    }

    pVcbuf->update(*pGfx,
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(width, height, 1.0f) * DirectX::XMMatrixTranslation(x, y, 0.0f) * pGfx->getView() * pGfx->getProjection()
        )
    );

    pVcbuf->bind(*pGfx);

    draw();
}