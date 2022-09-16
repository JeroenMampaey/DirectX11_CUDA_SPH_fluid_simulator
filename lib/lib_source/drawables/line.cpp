#include "line.h"

LineInitializerDesc::LineInitializerDesc(float x1, float y1, float x2, float y2, float red, float green, float blue) noexcept
    :
    x1(x1), 
    y1(y1), 
    x2(x2), 
    y2(y2), 
    red(red), 
    green(green), 
    blue(blue)
{}

Line::Line(DrawableInitializerDesc& desc, GraphicsEngine& gfx){
    LineInitializerDesc& castedDesc = static_cast<LineInitializerDesc&>(desc);
    x1 = castedDesc.x1; 
    y1 = castedDesc.y1; 
    x2 = castedDesc.x2; 
    y2 = castedDesc.y2;
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

void Line::initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const{
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
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    };

    sharedBinds.push_back(std::make_unique<VertexBuffer>(gfx, vertices));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(gfx, VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    sharedBinds.push_back(std::move(pvs));

    sharedBinds.push_back(std::make_unique<PixelShader>(gfx, PIXEL_PATH_CONCATINATED(L"PixelShader1.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 1
    };

    sharedBinds.push_back(std::make_unique<IndexBuffer>(gfx, indices));
    sharedIndexCount = indices.size();

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    sharedBinds.push_back(std::make_unique<InputLayout>(gfx, ied, pvsbc));

    sharedBinds.push_back(std::make_unique<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_LINELIST));
}

DirectX::XMMATRIX Line::getModel() const noexcept{
    return DirectX::XMMatrixScaling(x2-x1, y2-y1, 1.0f)*DirectX::XMMatrixTranslation(x1, y1, 0.0f);
}