#include "text.h"
#include <stdexcept>

#define NUM_USEFUL_ASCII_CHARS 96
#define FIRST_USEFUL_ASCII_CHAR 32
#define CHAR_PIXEL_WIDTH 8
#define CHAR_PIXEL_HEIGHT 12

DXTextInitializerDesc::DXTextInitializerDesc(float left_down_x, float left_down_y, float character_width, float character_height, std::string text) noexcept
    :
    left_down_x(left_down_x),
    left_down_y(left_down_y),
    character_width(character_width),
    character_height(character_height),
    text(text)
{}

DXText::DXText(DrawableInitializerDesc& desc, GraphicsEngine& gfx){
    DXTextInitializerDesc& castedDesc = static_cast<DXTextInitializerDesc&>(desc);
    left_down_x = castedDesc.left_down_x;
    left_down_y = castedDesc.left_down_y;
    character_width = castedDesc.character_width;
    character_height = castedDesc.character_height;
    text = castedDesc.text;

    struct Vertex
    {
        DirectX::XMFLOAT3 pos;
        struct
        {
            float u;
            float v;
        } tex;
    };

    std::vector<Vertex> vertices;
    std::vector<unsigned short> indices;

    if(text.size() > INT_MAX){
        throw std::overflow_error("data is larger than INT_MAX");
    }
    int stringLength = static_cast<int>(text.size());

    for (int i = 0; i < stringLength; i++) {
        const char& character = text[i];

        if(character < FIRST_USEFUL_ASCII_CHAR || character >= FIRST_USEFUL_ASCII_CHAR+NUM_USEFUL_ASCII_CHARS){
            throw std::exception("TextFactory received invalid characters");
        }

        vertices.push_back(Vertex{{(float)i, 0.0f, 0.0f}, {((float)(unsigned char)(character-FIRST_USEFUL_ASCII_CHAR))/((float)NUM_USEFUL_ASCII_CHARS), 0.0f}});
        vertices.push_back(Vertex{{(float)(i+1), 0.0f, 0.0f}, {((float)(unsigned char)(character+1-FIRST_USEFUL_ASCII_CHAR))/((float)NUM_USEFUL_ASCII_CHARS), 0.0f}});
        vertices.push_back(Vertex{{(float)i, 1.0f, 0.0f}, {((float)(unsigned char)(character-FIRST_USEFUL_ASCII_CHAR))/((float)NUM_USEFUL_ASCII_CHARS), 1.0f}});
        vertices.push_back(Vertex{{(float)(i+1), 1.0f, 0.0f}, {((float)(unsigned char)(character+1-FIRST_USEFUL_ASCII_CHAR))/((float)NUM_USEFUL_ASCII_CHARS), 1.0f}});
    
        indices.push_back(i*4);
        indices.push_back(i*4+2);
        indices.push_back(i*4+1);
        indices.push_back(i*4+1);
        indices.push_back(i*4+2);
        indices.push_back(i*4+3);
    }

    addUniqueBind(std::make_unique<IndexBuffer>(gfx, indices));
    setIndexCount(indices.size());

    addUniqueBind(std::make_unique<VertexBuffer>(gfx, vertices));

    addUniqueBind(std::make_unique<TransformCbufMVP>(gfx, *this));
}

void DXText::initializeSharedBinds(GraphicsEngine& gfx, std::vector<std::unique_ptr<Bindable>>& sharedBinds, int& sharedIndexCount) const{
    // copied from http://www.piclist.com/tecHREF/datafile/charset/extractor/charset_extractor.htm
    unsigned char bitmap[NUM_USEFUL_ASCII_CHARS*CHAR_PIXEL_HEIGHT] = {
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,	//  
        0x00,0x30,0x78,0x78,0x78,0x30,0x30,0x00,0x30,0x30,0x00,0x00,	// !
        0x00,0x66,0x66,0x66,0x24,0x00,0x00,0x00,0x00,0x00,0x00,0x00,	// "
        0x00,0x6C,0x6C,0xFE,0x6C,0x6C,0x6C,0xFE,0x6C,0x6C,0x00,0x00,	// #
        0x30,0x30,0x7C,0xC0,0xC0,0x78,0x0C,0x0C,0xF8,0x30,0x30,0x00,	// $
        0x00,0x00,0x00,0xC4,0xCC,0x18,0x30,0x60,0xCC,0x8C,0x00,0x00,	// %
        0x00,0x70,0xD8,0xD8,0x70,0xFA,0xDE,0xCC,0xDC,0x76,0x00,0x00,	// &
        0x00,0x30,0x30,0x30,0x60,0x00,0x00,0x00,0x00,0x00,0x00,0x00,	// '
        0x00,0x0C,0x18,0x30,0x60,0x60,0x60,0x30,0x18,0x0C,0x00,0x00,	// (
        0x00,0x60,0x30,0x18,0x0C,0x0C,0x0C,0x18,0x30,0x60,0x00,0x00,	// )
        0x00,0x00,0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00,0x00,0x00,	// *
        0x00,0x00,0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00,0x00,0x00,	// +
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x38,0x38,0x60,0x00,	// ,
        0x00,0x00,0x00,0x00,0x00,0xFE,0x00,0x00,0x00,0x00,0x00,0x00,	// -
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x38,0x38,0x00,0x00,	// .
        0x00,0x00,0x02,0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00,0x00,	// /
        0x00,0x7C,0xC6,0xD6,0xD6,0xD6,0xD6,0xD6,0xC6,0x7C,0x00,0x00,	// 0
        0x00,0x10,0x30,0xF0,0x30,0x30,0x30,0x30,0x30,0xFC,0x00,0x00,	// 1
        0x00,0x78,0xCC,0xCC,0x0C,0x18,0x30,0x60,0xCC,0xFC,0x00,0x00,	// 2
        0x00,0x78,0xCC,0x0C,0x0C,0x38,0x0C,0x0C,0xCC,0x78,0x00,0x00,	// 3
        0x00,0x0C,0x1C,0x3C,0x6C,0xCC,0xFE,0x0C,0x0C,0x1E,0x00,0x00,	// 4
        0x00,0xFC,0xC0,0xC0,0xC0,0xF8,0x0C,0x0C,0xCC,0x78,0x00,0x00,	// 5
        0x00,0x38,0x60,0xC0,0xC0,0xF8,0xCC,0xCC,0xCC,0x78,0x00,0x00,	// 6
        0x00,0xFE,0xC6,0xC6,0x06,0x0C,0x18,0x30,0x30,0x30,0x00,0x00,	// 7
        0x00,0x78,0xCC,0xCC,0xEC,0x78,0xDC,0xCC,0xCC,0x78,0x00,0x00,	// 8
        0x00,0x78,0xCC,0xCC,0xCC,0x7C,0x18,0x18,0x30,0x70,0x00,0x00,	// 9
        0x00,0x00,0x00,0x38,0x38,0x00,0x00,0x38,0x38,0x00,0x00,0x00,	// :
        0x00,0x00,0x00,0x38,0x38,0x00,0x00,0x38,0x38,0x18,0x30,0x00,	// ;
        0x00,0x0C,0x18,0x30,0x60,0xC0,0x60,0x30,0x18,0x0C,0x00,0x00,	// <
        0x00,0x00,0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,	// =
        0x00,0x60,0x30,0x18,0x0C,0x06,0x0C,0x18,0x30,0x60,0x00,0x00,	// >
        0x00,0x78,0xCC,0x0C,0x18,0x30,0x30,0x00,0x30,0x30,0x00,0x00,	// ?
        0x00,0x7C,0xC6,0xC6,0xDE,0xDE,0xDE,0xC0,0xC0,0x7C,0x00,0x00,	// @
        0x00,0x30,0x78,0xCC,0xCC,0xCC,0xFC,0xCC,0xCC,0xCC,0x00,0x00,	// A
        0x00,0xFC,0x66,0x66,0x66,0x7C,0x66,0x66,0x66,0xFC,0x00,0x00,	// B
        0x00,0x3C,0x66,0xC6,0xC0,0xC0,0xC0,0xC6,0x66,0x3C,0x00,0x00,	// C
        0x00,0xF8,0x6C,0x66,0x66,0x66,0x66,0x66,0x6C,0xF8,0x00,0x00,	// D
        0x00,0xFE,0x62,0x60,0x64,0x7C,0x64,0x60,0x62,0xFE,0x00,0x00,	// E
        0x00,0xFE,0x66,0x62,0x64,0x7C,0x64,0x60,0x60,0xF0,0x00,0x00,	// F
        0x00,0x3C,0x66,0xC6,0xC0,0xC0,0xCE,0xC6,0x66,0x3E,0x00,0x00,	// G
        0x00,0xCC,0xCC,0xCC,0xCC,0xFC,0xCC,0xCC,0xCC,0xCC,0x00,0x00,	// H
        0x00,0x78,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x78,0x00,0x00,	// I
        0x00,0x1E,0x0C,0x0C,0x0C,0x0C,0xCC,0xCC,0xCC,0x78,0x00,0x00,	// J
        0x00,0xE6,0x66,0x6C,0x6C,0x78,0x6C,0x6C,0x66,0xE6,0x00,0x00,	// K
        0x00,0xF0,0x60,0x60,0x60,0x60,0x62,0x66,0x66,0xFE,0x00,0x00,	// L
        0x00,0xC6,0xEE,0xFE,0xFE,0xD6,0xC6,0xC6,0xC6,0xC6,0x00,0x00,	// M
        0x00,0xC6,0xC6,0xE6,0xF6,0xFE,0xDE,0xCE,0xC6,0xC6,0x00,0x00,	// N
        0x00,0x38,0x6C,0xC6,0xC6,0xC6,0xC6,0xC6,0x6C,0x38,0x00,0x00,	// O
        0x00,0xFC,0x66,0x66,0x66,0x7C,0x60,0x60,0x60,0xF0,0x00,0x00,	// P
        0x00,0x38,0x6C,0xC6,0xC6,0xC6,0xCE,0xDE,0x7C,0x0C,0x1E,0x00,	// Q
        0x00,0xFC,0x66,0x66,0x66,0x7C,0x6C,0x66,0x66,0xE6,0x00,0x00,	// R
        0x00,0x78,0xCC,0xCC,0xC0,0x70,0x18,0xCC,0xCC,0x78,0x00,0x00,	// S
        0x00,0xFC,0xB4,0x30,0x30,0x30,0x30,0x30,0x30,0x78,0x00,0x00,	// T
        0x00,0xCC,0xCC,0xCC,0xCC,0xCC,0xCC,0xCC,0xCC,0x78,0x00,0x00,	// U
        0x00,0xCC,0xCC,0xCC,0xCC,0xCC,0xCC,0xCC,0x78,0x30,0x00,0x00,	// V
        0x00,0xC6,0xC6,0xC6,0xC6,0xD6,0xD6,0x6C,0x6C,0x6C,0x00,0x00,	// W
        0x00,0xCC,0xCC,0xCC,0x78,0x30,0x78,0xCC,0xCC,0xCC,0x00,0x00,	// X
        0x00,0xCC,0xCC,0xCC,0xCC,0x78,0x30,0x30,0x30,0x78,0x00,0x00,	// Y
        0x00,0xFE,0xCE,0x98,0x18,0x30,0x60,0x62,0xC6,0xFE,0x00,0x00,	// Z
        0x00,0x3C,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x3C,0x00,0x00,	// [
        0x00,0x00,0x80,0xC0,0x60,0x30,0x18,0x0C,0x06,0x02,0x00,0x00,	// slash in other direction (not '/' but the other one)
        0x00,0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00,0x00,	// ]
        0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,	// ^
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0x00,	// _
        0x30,0x30,0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,	// `
        0x00,0x00,0x00,0x00,0x78,0x0C,0x7C,0xCC,0xCC,0x76,0x00,0x00,	// a
        0x00,0xE0,0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0xDC,0x00,0x00,	// b
        0x00,0x00,0x00,0x00,0x78,0xCC,0xC0,0xC0,0xCC,0x78,0x00,0x00,	// c
        0x00,0x1C,0x0C,0x0C,0x7C,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00,	// d
        0x00,0x00,0x00,0x00,0x78,0xCC,0xFC,0xC0,0xCC,0x78,0x00,0x00,	// e
        0x00,0x38,0x6C,0x60,0x60,0xF8,0x60,0x60,0x60,0xF0,0x00,0x00,	// f
        0x00,0x00,0x00,0x00,0x76,0xCC,0xCC,0xCC,0x7C,0x0C,0xCC,0x78,	// g
        0x00,0xE0,0x60,0x60,0x6C,0x76,0x66,0x66,0x66,0xE6,0x00,0x00,	// h
        0x00,0x18,0x18,0x00,0x78,0x18,0x18,0x18,0x18,0x7E,0x00,0x00,	// i
        0x00,0x0C,0x0C,0x00,0x3C,0x0C,0x0C,0x0C,0x0C,0xCC,0xCC,0x78,	// j
        0x00,0xE0,0x60,0x60,0x66,0x6C,0x78,0x6C,0x66,0xE6,0x00,0x00,	// k
        0x00,0x78,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x00,	// l
        0x00,0x00,0x00,0x00,0xFC,0xD6,0xD6,0xD6,0xD6,0xC6,0x00,0x00,	// m
        0x00,0x00,0x00,0x00,0xF8,0xCC,0xCC,0xCC,0xCC,0xCC,0x00,0x00,	// n
        0x00,0x00,0x00,0x00,0x78,0xCC,0xCC,0xCC,0xCC,0x78,0x00,0x00,	// o
        0x00,0x00,0x00,0x00,0xDC,0x66,0x66,0x66,0x66,0x7C,0x60,0xF0,	// p
        0x00,0x00,0x00,0x00,0x76,0xCC,0xCC,0xCC,0xCC,0x7C,0x0C,0x1E,	// q
        0x00,0x00,0x00,0x00,0xEC,0x6E,0x76,0x60,0x60,0xF0,0x00,0x00,	// r
        0x00,0x00,0x00,0x00,0x78,0xCC,0x60,0x18,0xCC,0x78,0x00,0x00,	// s
        0x00,0x00,0x20,0x60,0xFC,0x60,0x60,0x60,0x6C,0x38,0x00,0x00,	// t
        0x00,0x00,0x00,0x00,0xCC,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00,	// u
        0x00,0x00,0x00,0x00,0xCC,0xCC,0xCC,0xCC,0x78,0x30,0x00,0x00,	// v
        0x00,0x00,0x00,0x00,0xC6,0xC6,0xD6,0xD6,0x6C,0x6C,0x00,0x00,	// w
        0x00,0x00,0x00,0x00,0xC6,0x6C,0x38,0x38,0x6C,0xC6,0x00,0x00,	// x
        0x00,0x00,0x00,0x00,0x66,0x66,0x66,0x66,0x3C,0x0C,0x18,0xF0,	// y
        0x00,0x00,0x00,0x00,0xFC,0x8C,0x18,0x60,0xC4,0xFC,0x00,0x00,	// z
        0x00,0x1C,0x30,0x30,0x60,0xC0,0x60,0x30,0x30,0x1C,0x00,0x00,	// {
        0x00,0x18,0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x18,0x00,0x00,	// |
        0x00,0xE0,0x30,0x30,0x18,0x0C,0x18,0x30,0x30,0xE0,0x00,0x00,	// }
        0x00,0x73,0xDA,0xCE,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,	// ~
        0x00,0x00,0x00,0x10,0x38,0x6C,0xC6,0xC6,0xFE,0x00,0x00,0x00     // DEL
    };
    std::unique_ptr<Texture::Color[]> textureBuffer = std::make_unique<Texture::Color[]>(NUM_USEFUL_ASCII_CHARS*CHAR_PIXEL_HEIGHT*CHAR_PIXEL_WIDTH);
    for(int i=0; i<NUM_USEFUL_ASCII_CHARS; i++){
        for(int j=0; j<CHAR_PIXEL_HEIGHT; j++){
            for(int e=0; e<CHAR_PIXEL_WIDTH; e++){
                if((bitmap[i*CHAR_PIXEL_HEIGHT+j] << e) & 0b10000000){
                    textureBuffer[(CHAR_PIXEL_HEIGHT-1-j)*NUM_USEFUL_ASCII_CHARS*CHAR_PIXEL_WIDTH+i*CHAR_PIXEL_WIDTH+e] = {255, 255, 255, 255};
                }
                else{
                    textureBuffer[(CHAR_PIXEL_HEIGHT-1-j)*NUM_USEFUL_ASCII_CHARS*CHAR_PIXEL_WIDTH+i*CHAR_PIXEL_WIDTH+e] = {255, 255, 255, 0};
                }
            }
        }
    }

    sharedBinds.push_back(std::make_unique<Texture>(gfx, std::move(textureBuffer), NUM_USEFUL_ASCII_CHARS*CHAR_PIXEL_WIDTH, CHAR_PIXEL_HEIGHT));

    sharedBinds.push_back(std::make_unique<Sampler>(gfx, D3D11_FILTER_MIN_MAG_MIP_POINT));

    std::unique_ptr<VertexShader> pvs = std::make_unique<VertexShader>(gfx, VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    sharedBinds.push_back(std::move(pvs));

    sharedBinds.push_back(std::make_unique<PixelShader>(gfx, PIXEL_PATH_CONCATINATED(L"PixelShader3.cso")));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        { "Position",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0 },
        { "TexCoord",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0 },
    };
    sharedBinds.push_back(std::make_unique<InputLayout>(gfx, ied, pvsbc));

    sharedBinds.push_back(std::make_unique<Topology>(gfx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
}

DirectX::XMMATRIX DXText::getModel() const noexcept{
    return DirectX::XMMatrixScaling(character_width, character_height, 1.0f)*DirectX::XMMatrixTranslation(left_down_x, left_down_y, 0.0f);
}