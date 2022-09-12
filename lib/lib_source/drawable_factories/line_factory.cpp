#include "line_factory.h"

LineStateUpdateDesc::LineStateUpdateDesc(float x1, float y1, float x2, float y2) noexcept
    : 
    new_x1(x1), 
    new_y1(y1), 
    new_x2(x2), 
    new_y2(y2)
{}

LineStateInitializerDesc::LineStateInitializerDesc(float x1, float y1, float x2, float y2, float red, float green, float blue) noexcept
    :
    x1(x1), 
    y1(y1), 
    x2(x2), 
    y2(y2), 
    red(red), 
    green(green), 
    blue(blue)
{}

LineState::LineState(float x1, float y1, float x2, float y2, float red, float green, float blue) noexcept
    :
    x1(x1), 
    y1(y1), 
    x2(x2), 
    y2(y2), 
    red(red), 
    green(green), 
    blue(blue)
{}

DirectX::XMMATRIX LineState::getTransformXM() const noexcept{
    return DirectX::XMMatrixScaling(x2-x1, y2-y1, 1.0f)*DirectX::XMMatrixTranslation(x1, y1, 0.0f);
}

void LineState::update(DrawableStateUpdateDesc& desc) noexcept{
    LineStateUpdateDesc& castedDesc = static_cast<LineStateUpdateDesc&>(desc);
    x1 = castedDesc.new_x1;
    y1 = castedDesc.new_y1;
    x2 = castedDesc.new_x2;
    y2 = castedDesc.new_y2;
}

std::unique_ptr<DrawableState> LineFactory::getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept{
    LineStateInitializerDesc& castedDesc = static_cast<LineStateInitializerDesc&>(desc);
    return std::make_unique<LineState>(castedDesc.x1, castedDesc.y1, castedDesc.x2, castedDesc.y2, castedDesc.red, castedDesc.green, castedDesc.blue);
}

void LineFactory::initializeSharedBinds(GraphicsEngine& gfx){
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

    addSharedBind(std::make_shared<VertexBuffer>(gfx, vertices));

    std::shared_ptr<VertexShader> pvs = std::make_shared<VertexShader>(gfx, VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(pvs);

    addSharedBind(std::make_shared<PixelShader>(gfx, PIXEL_PATH_CONCATINATED(L"PixelShader1.cso")));

    const std::vector<unsigned short> indices = 
    {
        0, 1
    };

    addSharedBind(std::make_shared<IndexBuffer>(gfx, indices));
    setSharedIndexCount(indices.size());

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(std::make_shared<InputLayout>(gfx, ied, pvsbc));

    addSharedBind(std::make_shared<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_LINELIST));
}

void LineFactory::initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const{
    LineState& state = static_cast<LineState&>(drawable.getState());
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
