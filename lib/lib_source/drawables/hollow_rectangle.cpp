#include "hollow_rectangle.h"

HollowRectangleInitializerDesc::HollowRectangleInitializerDesc(float x, float y, float width, float height, float red, float green, float blue) noexcept
    :
    x(x),
    y(y),
    width(width),
    height(height),
    red(red),
    green(green),
    blue(blue)
{}

HollowRectangle::HollowRectangle(DrawableInitializerDesc& desc, GraphicsEngine& gfx){
    HollowRectangleInitializerDesc& castedDesc = static_cast<HollowRectangleInitializerDesc&>(desc);
    x = castedDesc.x;
    y = castedDesc.y;
    width = castedDesc.width;
    height = castedDesc.height;
    red = castedDesc.red;
    green = castedDesc.green;
    blue = castedDesc.blue;

    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {red, green, blue};

    addUniqueBind(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(0, gfx, cb2));

    addUniqueBind(std::make_unique<TransformCbufMVP>(gfx, *this));
}

void HollowRectangle::initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const{
    struct Vertex
    {
        struct
        {
            float x;
            float y;
            float z;
        } pos;
    };

    const std::vector<Vertex> vertices =
    {
        {-0.5f, -0.5f, 0.0f},
        {0.5f, -0.5f, 0.0f},
        {-0.5f, 0.5f, 0.0f},
        {0.5f, 0.5f, 0.0f}
    };

    sharedBinds.push_back(std::make_unique<VertexBuffer>(gfx, vertices));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(gfx, VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    sharedBinds.push_back(std::move(pvs));

    sharedBinds.push_back(std::make_unique<PixelShader>(gfx, PIXEL_PATH_CONCATINATED(L"PixelShader1.cso")));

    const std::vector<unsigned short> indices = 
    {
        0,
        1,
        3,
        2,
        0
    };

    sharedBinds.push_back(std::make_unique<IndexBuffer>(gfx, indices));
    sharedIndexCount = indices.size();

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    sharedBinds.push_back(std::make_unique<InputLayout>(gfx, ied, pvsbc));

    sharedBinds.push_back(std::make_unique<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP));
}

DirectX::XMMATRIX HollowRectangle::getModel() const noexcept{
    return DirectX::XMMatrixScaling(width, height, 1.0f)*DirectX::XMMatrixTranslation(x, y, 0.0f);
}