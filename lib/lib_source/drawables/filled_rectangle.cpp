#include "filled_rectangle.h"

FilledRectangleInitializerDesc::FilledRectangleInitializerDesc(float x, float y, float width, float height) noexcept
    :
    x(x), 
    y(y),
    width(width),
    height(height)
{}

FilledRectangle::FilledRectangle(DrawableInitializerDesc& desc, GraphicsEngine& gfx){
    FilledRectangleInitializerDesc& castedDesc = static_cast<FilledRectangleInitializerDesc&>(desc);
    x = castedDesc.x;
    y = castedDesc.y;
    width = castedDesc.width;
    height = castedDesc.height;

    addUniqueBind(std::make_unique<TransformCbuf>(gfx, *this));
}

void FilledRectangle::initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const{
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
    const ConstantBuffer2 cb2 = {1.0f, 0.0f, 1.0f};

    sharedBinds.push_back(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(gfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    sharedBinds.push_back(std::make_unique<InputLayout>(gfx, ied, pvsbc));

    sharedBinds.push_back(std::make_unique<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
}

DirectX::XMMATRIX FilledRectangle::getTransformXM() const noexcept{
    return DirectX::XMMatrixScaling(width, height, 1.0f)*DirectX::XMMatrixTranslation(x, y, 0.0f);
}
