#include "filled_circle.h"
#include "drawable_includes.h"

DXFilledCircleInitializerDesc::DXFilledCircleInitializerDesc(float x, float y, float radius) noexcept
    :
    x(x),
    y(y),
    radius(radius)
{}

DXFilledCircle::DXFilledCircle(DrawableInitializerDesc& desc, GraphicsEngine& gfx){
    DXFilledCircleInitializerDesc castedDesc = static_cast<DXFilledCircleInitializerDesc&>(desc);
    x = castedDesc.x;
    y = castedDesc.y;
    radius = castedDesc.radius;

    addUniqueBind(std::make_unique<TransformCbufMVP>(gfx, *this));
}

void DXFilledCircle::initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const{
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

    sharedBinds.push_back(std::make_unique<VertexBuffer>(gfx, vertices));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(gfx, VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    sharedBinds.push_back(std::move(pvs));

    sharedBinds.push_back(std::make_unique<PixelShader>(gfx, PIXEL_PATH_CONCATINATED(L"PixelShader2.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 2, 1,
        1, 2, 3
    };

    sharedBinds.push_back(std::make_unique<IndexBuffer>(gfx, indices));
    sharedIndexCount = indices.size();

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {0.2f, 0.2f, 1.0f};

    sharedBinds.push_back(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(0, gfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        { "Position",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0 },
        { "TexCoord",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0 },
    };
    sharedBinds.push_back(std::make_unique<InputLayout>(gfx, ied, pvsbc));

    sharedBinds.push_back(std::make_unique<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
}

DirectX::XMMATRIX DXFilledCircle::getModel() const noexcept{
    return DirectX::XMMatrixScaling(radius, radius, 1.0f)*DirectX::XMMatrixTranslation(x, y, 0.0f);
}