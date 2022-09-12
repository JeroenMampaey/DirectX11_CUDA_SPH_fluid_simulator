#include "hollow_rectangle_factory.h"

HollowRectangleStateUpdateDesc::HollowRectangleStateUpdateDesc(float new_x, float new_y) noexcept
    :
    new_x(new_x),
    new_y(new_y)
{}

HollowRectangleStateInitializerDesc::HollowRectangleStateInitializerDesc(float x, float y, float width, float height, float red, float green, float blue) noexcept
    :
    x(x),
    y(y),
    width(width),
    height(height),
    red(red),
    green(green),
    blue(blue)
{}

HollowRectangleState::HollowRectangleState(float x, float y, float width, float height, float red, float green, float blue) noexcept
    :
    x(x),
    y(y),
    width(width),
    height(height),
    red(red),
    green(green),
    blue(blue)
{}

DirectX::XMMATRIX HollowRectangleState::getTransformXM() const noexcept{
    return DirectX::XMMatrixScaling(width, height, 1.0f)*DirectX::XMMatrixTranslation(x, y, 0.0f);
}

void HollowRectangleState::update(DrawableStateUpdateDesc& desc) noexcept{
    HollowRectangleStateUpdateDesc& castedDesc = static_cast<HollowRectangleStateUpdateDesc&>(desc);
    x = castedDesc.new_x;
    y = castedDesc.new_y;
}

std::unique_ptr<DrawableState> HollowRectangleFactory::getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept{
    HollowRectangleStateInitializerDesc& castedDesc = static_cast<HollowRectangleStateInitializerDesc&>(desc);
    return std::make_unique<HollowRectangleState>(castedDesc.x, castedDesc.y, castedDesc.width, castedDesc.height, castedDesc.red, castedDesc.green, castedDesc.blue);
}

void HollowRectangleFactory::initializeSharedBinds(GraphicsEngine& gfx){
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
        0,
        1,
        3,
        2,
        0
    };

    addSharedBind(std::make_shared<IndexBuffer>(gfx, indices));
    setSharedIndexCount(indices.size());

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(std::make_shared<InputLayout>(gfx, ied, pvsbc));

    addSharedBind(std::make_shared<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP));
}

void HollowRectangleFactory::initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const{
    HollowRectangleState& state = static_cast<HollowRectangleState&>(drawable.getState());
    struct ConstantBuffer2
    {
        float r;
        float g;
        float b;
        float a;
    };
    const ConstantBuffer2 cb2 = {state.red, state.green, state.blue};

    addUniqueBind(drawable, std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(gfx, cb2));

    addUniqueBind(drawable, std::make_unique<TransformCbuf>(gfx));
}