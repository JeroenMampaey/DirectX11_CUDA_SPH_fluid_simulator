#include "square.h"
#include "../bindables/transformcbuf.h"
#include "../bindables/vertexbuffer.h"
#include "../bindables/vertexshader.h"
#include "../bindables/pixelshader.h"
#include "../bindables/indexbuffer.h"
#include "../bindables/inputlayout.h"
#include "../bindables/topology.h"
#include <memory>

Square::Square(GraphicsEngine& gfx, float width, float height, float x, float y) 
    : 
    Drawable<Square>(gfx), 
    width(width),
    height(height),
    x(x),
    y(y)
{
    initializeDrawable(gfx);
    AddBind(std::make_unique<TransformCbuf<Square>>(gfx,*this));
}

DirectX::XMMATRIX Square::GetTransformXM() const noexcept{
    return DirectX::XMMatrixScaling(width, height, 1.0f)*DirectX::XMMatrixTranslation(x, y, 0.0f) * DirectX::XMMatrixTranslation(0.0f, 0.0f, 5.0f);
}

void Square::update(float new_x, float new_y) noexcept{
    x = new_x;
    y = new_y;
}

float Square::getX() const noexcept{
    return x;
}

float Square::getY() const noexcept{
    return y;
}

void Square::createSharedBinds(GraphicsEngine& gfx){
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

    AddSharedBind(std::make_unique<VertexBuffer>(gfx, vertices));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(gfx, L"VertexShader.cso");  //TODO!!!!
	ID3DBlob* pvsbc = pvs->GetBytecode();
    AddSharedBind(std::move(pvs));

    AddSharedBind(std::make_unique<PixelShader>(gfx, L"PixelShader.cso"));  //TODO!!!!

    const std::vector<unsigned short> indices = 
    {
        0, 2, 1,
        1, 2, 3
    };

    AddSharedBind(std::make_unique<IndexBuffer>(gfx, indices));
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

    AddSharedBind(std::make_unique<PixelConstantBuffer<ConstantBuffer2>>(gfx, cb2));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    AddSharedBind(std::make_unique<InputLayout>(gfx, ied, pvsbc));

    AddSharedBind(std::make_unique<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
}