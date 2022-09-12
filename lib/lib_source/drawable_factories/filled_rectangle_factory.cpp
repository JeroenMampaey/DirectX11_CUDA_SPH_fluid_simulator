#include "filled_rectangle_factory.h"

FilledRectangleStateUpdateDesc::FilledRectangleStateUpdateDesc(float new_x, float new_y) noexcept : new_x(new_x), new_y(new_y){}

FilledRectangleStateInitializerDesc::FilledRectangleStateInitializerDesc(float x, float y, float width, float height) noexcept 
    : 
    x(x), 
    y(y),
    width(width),
    height(height)
{}

FilledRectangleState::FilledRectangleState(float x, float y, float width, float height) noexcept 
    : 
    x(x),
    y(y),
    width(width),
    height(height)
{}

DirectX::XMMATRIX FilledRectangleState::getTransformXM() const noexcept{
    return DirectX::XMMatrixScaling(width, height, 1.0f)*DirectX::XMMatrixTranslation(x, y, 0.0f);
}

void FilledRectangleState::update(DrawableStateUpdateDesc& desc) noexcept{
    FilledRectangleStateUpdateDesc& castedDesc = static_cast<FilledRectangleStateUpdateDesc&>(desc);
    x = castedDesc.new_x;
    y = castedDesc.new_y;
}

std::unique_ptr<DrawableState> FilledRectangleFactory::getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept{
    FilledRectangleStateInitializerDesc& castedDesc = static_cast<FilledRectangleStateInitializerDesc&>(desc);
    return std::make_unique<FilledRectangleState>(castedDesc.x, castedDesc.y, castedDesc.width, castedDesc.height);
}


void FilledRectangleFactory::initializeSharedBinds(GraphicsEngine& gfx){
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

    addSharedBind(std::make_shared<VertexBuffer>(gfx, vertices));

    std::shared_ptr<VertexShader> pvs = std::make_shared<VertexShader>(gfx, VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(pvs);

    addSharedBind(std::make_shared<PixelShader>(gfx, PIXEL_PATH_CONCATINATED(L"PixelShader1.cso")));

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
    const ConstantBuffer2 cb2 = {1.0f, 0.0f, 1.0f};

    addSharedBind(std::make_shared<PixelConstantBuffer<ConstantBuffer2>>(gfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(std::make_shared<InputLayout>(gfx, ied, pvsbc));

    addSharedBind(std::make_shared<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
}

void FilledRectangleFactory::initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const{
    addUniqueBind(drawable, std::make_unique<TransformCbuf>(gfx));
}
