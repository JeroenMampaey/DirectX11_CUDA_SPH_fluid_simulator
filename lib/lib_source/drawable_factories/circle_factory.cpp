#include "circle_factory.h"

#define CIRCLE_IMAGE_RADIUS 20

CircleStateUpdateDesc::CircleStateUpdateDesc(float new_x, float new_y) noexcept
    :
    new_x(new_x),
    new_y(new_y)
{}

CircleStateInitializerDesc::CircleStateInitializerDesc(float x, float y, float radius) noexcept
    :
    x(x),
    y(y),
    radius(radius)
{}

CircleState::CircleState(float x, float y, float radius) noexcept
    :
    x(x),
    y(y),
    radius(radius)
{}

DirectX::XMMATRIX CircleState::getTransformXM() const noexcept{
    return DirectX::XMMatrixScaling(radius, radius, 1.0f)*DirectX::XMMatrixTranslation(x, y, 0.0f);
}


void CircleState::update(DrawableStateUpdateDesc& desc) noexcept{
    CircleStateUpdateDesc& castedDesc = static_cast<CircleStateUpdateDesc&>(desc);
    x = castedDesc.new_x;
    y = castedDesc.new_y;
}

std::unique_ptr<DrawableState> CircleFactory::getInitialDrawableState(DrawableStateInitializerDesc& desc) const noexcept{
    CircleStateInitializerDesc& castedDesc = static_cast<CircleStateInitializerDesc&>(desc);
    return std::make_unique<CircleState>(castedDesc.x, castedDesc.y, castedDesc.radius);
}

void CircleFactory::initializeSharedBinds(GraphicsEngine& gfx){
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
        {{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
        {{1.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
        {{1.0f, 1.0f, 0.0f}, {1.0f, 1.0f}}
    };

    std::unique_ptr<Texture::Color[]> textureBuffer = std::make_unique<Texture::Color[]>((2*CIRCLE_IMAGE_RADIUS)*(2*CIRCLE_IMAGE_RADIUS));
    for(int y_index=0; y_index<2*CIRCLE_IMAGE_RADIUS; y_index++){
        for(int x_index=0; x_index<2*CIRCLE_IMAGE_RADIUS; x_index++){
            float x = x_index+0.5f;
            float y = y_index+0.5f;
            if((x-CIRCLE_IMAGE_RADIUS)*(x-CIRCLE_IMAGE_RADIUS)+(y-CIRCLE_IMAGE_RADIUS)*(y-CIRCLE_IMAGE_RADIUS) < CIRCLE_IMAGE_RADIUS*CIRCLE_IMAGE_RADIUS){
                textureBuffer[y_index*2*CIRCLE_IMAGE_RADIUS+x_index] = {255, 51, 51, 255};
            }
            else{
                textureBuffer[y_index*2*CIRCLE_IMAGE_RADIUS+x_index] = {0, 51, 51, 255};
            }
        }
    }
    addSharedBind(std::make_shared<Texture>(gfx, std::move(textureBuffer), 2*CIRCLE_IMAGE_RADIUS, 2*CIRCLE_IMAGE_RADIUS));

    addSharedBind(std::make_shared<VertexBuffer>(gfx, vertices));

    addSharedBind(std::make_shared<Sampler>(gfx));

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

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        { "Position",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0 },
        { "TexCoord",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0 },
    };
    addSharedBind(std::make_shared<InputLayout>(gfx, ied, pvsbc));

    addSharedBind(std::make_shared<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
}

void CircleFactory::initializeBinds(GraphicsEngine& gfx, Drawable& drawable) const{
    addUniqueBind(drawable, std::make_unique<TransformCbuf>(gfx));
}