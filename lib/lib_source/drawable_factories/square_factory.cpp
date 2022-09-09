#include "square_factory.h"

SquareStateUpdateDesc::SquareStateUpdateDesc(float new_x, float new_y) noexcept 
    : 
    DrawableStateUpdateDesc(), 
    new_x(new_x), 
    new_y(new_y)
{}

SquareState::SquareState(float x, float y) noexcept : DrawableState(), x(x), y(y){}

DirectX::XMMATRIX SquareState::getTransformXM() const noexcept{
    return DirectX::XMMatrixTranslation(x, y, 0.0f) * DirectX::XMMatrixTranslation(0.0f, 0.0f, 5.0f);
}

void SquareState::update(DrawableStateUpdateDesc& desc) noexcept{
    SquareStateUpdateDesc& castedDesc = static_cast<SquareStateUpdateDesc&>(desc);
    x = castedDesc.new_x;
    y = castedDesc.new_y;
}

std::unique_ptr<DrawableState> SquareFactory::getInitialDrawableState() const noexcept{
    return std::make_unique<SquareState>(0.0f, 0.0f);
}


void SquareFactory::initializeSharedBinds(GraphicsEngine& gfx){
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
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}
    };

    addSharedBind(std::make_shared<VertexBuffer>(gfx, vertices));

    std::shared_ptr<VertexShader> pvs = std::make_shared<VertexShader>(gfx, VERTEX_PATH_CONCATINATED(L"VertexShader1.cso"));
    ID3DBlob* pvsbc = pvs->GetBytecode();
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
        struct
        {
            float r;
            float g;
            float b;
            float a;
        } face_colors[6];
    };
    const ConstantBuffer2 cb2 =
    {
        {
            {1.0f, 0.0f, 1.0f},
            {1.0f, 0.0f, 1.0f}
        }
    };

    addSharedBind(std::make_shared<PixelConstantBuffer<ConstantBuffer2>>(gfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(std::make_shared<InputLayout>(gfx, ied, pvsbc));

    addSharedBind(std::make_shared<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
}

void SquareFactory::initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const{
    addUniqueBind(drawable, std::make_unique<TransformCbuf>(gfx));
}
