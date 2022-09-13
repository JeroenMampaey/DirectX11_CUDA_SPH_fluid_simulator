#include "filled_circle_factory.h"

FilledCircleStateInitializerDesc::FilledCircleStateInitializerDesc(float x, float y, float radius) noexcept
    :
    x(x),
    y(y),
    radius(radius)
{}

FilledCircleState::FilledCircleState(float x, float y, float radius) noexcept
    :
    x(x),
    y(y),
    radius(radius)
{}

DirectX::XMMATRIX FilledCircleState::getTransformXM() const noexcept{
    return DirectX::XMMatrixScaling(radius, radius, 1.0f)*DirectX::XMMatrixTranslation(x, y, 0.0f);
}

std::unique_ptr<DrawableState> FilledCircleFactory::getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept{
    FilledCircleStateInitializerDesc& castedDesc = static_cast<FilledCircleStateInitializerDesc&>(desc);
    return std::make_unique<FilledCircleState>(castedDesc.x, castedDesc.y, castedDesc.radius);
}

void FilledCircleFactory::initializeSharedBinds(GraphicsEngine& gfx){
    struct Vertex
    {
        DirectX::XMFLOAT3 pos;
        struct
        {
            float u;
            float v;
        } tex;
    };

    const std::vector<Vertex> vertices =
    {
        {{-1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f}},
        {{1.0f, -1.0f, 0.0f}, {1.0f, -1.0f}},
        {{-1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f}},
        {{1.0f, 1.0f, 0.0f}, {1.0f, 1.0f}}
    };

    addSharedBind(std::make_shared<VertexBuffer>(gfx, vertices));

    std::shared_ptr<VertexShader> pvs = std::make_shared<VertexShader>(gfx, VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(pvs);

    addSharedBind(std::make_shared<PixelShader>(gfx, PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 2, 1,
        1, 2, 3
    };

    addSharedBind(std::make_shared<IndexBuffer>(gfx, indices));
    setSharedIndexCount(indices.size());

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {0.2f, 0.2f, 1.0f};

    addSharedBind(std::make_shared<PixelConstantBuffer<ConstantBuffer2>>(gfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        { "Position",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0 },
        { "TexCoord",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0 },
    };
    addSharedBind(std::make_shared<InputLayout>(gfx, ied, pvsbc));

    addSharedBind(std::make_shared<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
}

void FilledCircleFactory::initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const{
    addUniqueBind(drawable, std::make_unique<TransformCbuf>(gfx));
}