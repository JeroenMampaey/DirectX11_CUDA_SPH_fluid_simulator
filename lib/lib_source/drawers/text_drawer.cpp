#include "text_drawer.h"
#include <stdexcept>
#include <map>
#include "../helpers.h"
#include "../bindables/bindables_includes.h"

#define VERTICES_PER_CHARACTER 4
#define INDICES_PER_CHARACTER 6

#define NUM_USEFUL_ASCII_CHARS 96
#define FIRST_USEFUL_ASCII_CHAR 32
#define CHAR_PIXEL_WIDTH 8
#define CHAR_PIXEL_HEIGHT 12

DynamicTextDrawer::~DynamicTextDrawer() noexcept = default;
StaticScreenTextDrawer::~StaticScreenTextDrawer() noexcept = default;

DynamicTextDrawer::DynamicTextDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, float red, float green, float blue)
    :
    Drawer(pDrawerHelper)
{
    GraphicsEngine& gfx = helper->getGraphicsEngine();

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
                    textureBuffer[(CHAR_PIXEL_HEIGHT-1-j)*NUM_USEFUL_ASCII_CHARS*CHAR_PIXEL_WIDTH+i*CHAR_PIXEL_WIDTH+e] = {(unsigned char)(blue*255.0f), (unsigned char)(green*255.0f), (unsigned char)(red*255.0f), 255};
                }
                else{
                    textureBuffer[(CHAR_PIXEL_HEIGHT-1-j)*NUM_USEFUL_ASCII_CHARS*CHAR_PIXEL_WIDTH+i*CHAR_PIXEL_WIDTH+e] = {(unsigned char)(blue*255.0f), (unsigned char)(green*255.0f), (unsigned char)(red*255.0f), 0};
                }
            }
        }
    }
    addSharedBind(gfx.createNewGraphicsBoundObject<Texture>(std::move(textureBuffer), NUM_USEFUL_ASCII_CHARS*CHAR_PIXEL_WIDTH, CHAR_PIXEL_HEIGHT));

    addSharedBind(gfx.createNewGraphicsBoundObject<Sampler>(D3D11_FILTER_MIN_MAG_MIP_POINT));

    std::unique_ptr<VertexShader> pvs = gfx.createNewGraphicsBoundObject<VertexShader>(VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(gfx.createNewGraphicsBoundObject<PixelShader>(PIXEL_PATH_CONCATINATED(L"PixelShader3.cso")));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(gfx.createNewGraphicsBoundObject<InputLayout>(ied, pvsbc));

    addSharedBind(gfx.createNewGraphicsBoundObject<Topology>(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    pVcbuf = gfx.createNewGraphicsBoundObject<VertexConstantBuffer<DirectX::XMMATRIX>>(0);

    std::vector<DirectX::XMFLOAT3> vertices;
    std::vector<DirectX::XMFLOAT2> texturecoords;
    std::vector<unsigned short> indices;

    for (int i = 0; i < MAX_DYNAMIC_TEXT_DRAWER_STRLEN; i++){
        vertices.push_back({(float)i, 0.0f, 0.0f});
        vertices.push_back({(float)(i+1), 0.0f, 0.0f});
        vertices.push_back({(float)i, 1.0f, 0.0f});
        vertices.push_back({(float)(i+1), 1.0f, 0.0f});

        texturecoords.push_back({0.0f, 0.0f});
        texturecoords.push_back({0.0f, 0.0f});
        texturecoords.push_back({0.0f, 0.0f});
        texturecoords.push_back({0.0f, 0.0f});
    
        indices.push_back(i*VERTICES_PER_CHARACTER);
        indices.push_back(i*VERTICES_PER_CHARACTER+2);
        indices.push_back(i*VERTICES_PER_CHARACTER+1);
        indices.push_back(i*VERTICES_PER_CHARACTER+1);
        indices.push_back(i*VERTICES_PER_CHARACTER+2);
        indices.push_back(i*VERTICES_PER_CHARACTER+3);
    }

    const void* vertexBuffers[] = {(void*)vertices.data(), (void*)texturecoords.data()};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3), sizeof(DirectX::XMFLOAT2)};
    const UINT cpuAccessFlags[] = {0, D3D11_CPU_ACCESS_WRITE};
    const size_t numVertices[] = {VERTICES_PER_CHARACTER*MAX_DYNAMIC_TEXT_DRAWER_STRLEN, VERTICES_PER_CHARACTER*MAX_DYNAMIC_TEXT_DRAWER_STRLEN};
    pVBuf = gfx.createNewGraphicsBoundObject<CpuMappableVertexBuffer>(vertexBuffers, vertexSizes, cpuAccessFlags, numVertices, 2);
    addSharedBind(gfx.createNewGraphicsBoundObject<IndexBuffer>(indices));
}

void DynamicTextDrawer::drawDynamicText(const std::string& text, float left_down_x, float left_down_y, float char_width, float char_height){
    if(text.size()==0){
        return;
    }

    if(text.size()>MAX_DYNAMIC_TEXT_DRAWER_STRLEN){
        throw std::overflow_error("DynamicTextDrawer can only draw up to "+std::to_string(MAX_DYNAMIC_TEXT_DRAWER_STRLEN)+" characters");
    }

    if(text.size() > INT_MAX){
        throw std::overflow_error("DynamicTextDrawer was called with a text with length bigger than INT_MAX");
    }

    unsigned int stringLength = static_cast<int>(text.size());

    setIndexCount(stringLength*INDICES_PER_CHARACTER);

    DirectX::XMFLOAT2* newTextureCoords = static_cast<DirectX::XMFLOAT2*>(pVBuf->getMappedAccess(1));

    for(int i=0; i<stringLength; i++){
        const char& character = text[i];

        if(character < FIRST_USEFUL_ASCII_CHAR || character >= FIRST_USEFUL_ASCII_CHAR+NUM_USEFUL_ASCII_CHARS){
            throw std::exception("DynamicTextDrawer received invalid characters");
        }

        newTextureCoords[i*VERTICES_PER_CHARACTER] = {((float)(unsigned char)(character-FIRST_USEFUL_ASCII_CHAR))/((float)NUM_USEFUL_ASCII_CHARS), 0.0f};
        newTextureCoords[i*VERTICES_PER_CHARACTER+1] = {((float)(unsigned char)(character+1-FIRST_USEFUL_ASCII_CHAR))/((float)NUM_USEFUL_ASCII_CHARS), 0.0f};
        newTextureCoords[i*VERTICES_PER_CHARACTER+2] = {((float)(unsigned char)(character-FIRST_USEFUL_ASCII_CHAR))/((float)NUM_USEFUL_ASCII_CHARS), 1.0f};
        newTextureCoords[i*VERTICES_PER_CHARACTER+3] = {((float)(unsigned char)(character+1-FIRST_USEFUL_ASCII_CHAR))/((float)NUM_USEFUL_ASCII_CHARS), 1.0f};
    }

    pVBuf->unMap(1);

    pVBuf->bind();

    GraphicsEngine& gfx = helper->getGraphicsEngine();
    pVcbuf->update(
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(char_width, char_height, 1.0f) * DirectX::XMMatrixTranslation(left_down_x, left_down_y, 0.0f) * gfx.getView() * gfx.getProjection()
        )
    );
    pVcbuf->bind();

    bindSharedBinds(typeid(DynamicTextDrawer));
    drawIndexed();
}

StaticScreenTextDrawer::StaticScreenTextDrawer(std::shared_ptr<DrawerHelper> pDrawerHelper, const std::string& text, float left_down_x, float left_down_y, float char_width, float char_height, float red, float green, float blue)
    :
    Drawer(pDrawerHelper)
{
    if(text.size()==0){
        return;
    }

    if(text.size() > INT_MAX){
        throw std::overflow_error("StaticScreenTextDrawer was called with a text with length bigger than INT_MAX");
    }

    GraphicsEngine& gfx = helper->getGraphicsEngine();

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

    addSharedBind(gfx.createNewGraphicsBoundObject<Sampler>( D3D11_FILTER_MIN_MAG_MIP_POINT));

    std::unique_ptr<VertexShader> pvs = gfx.createNewGraphicsBoundObject<VertexShader>(VERTEX_PATH_CONCATINATED(L"VertexShader2.cso"));
    ID3DBlob* pvsbc = pvs->getBytecode();
    addSharedBind(std::move(pvs));

    addSharedBind(gfx.createNewGraphicsBoundObject<PixelShader>(PIXEL_PATH_CONCATINATED(L"PixelShader3.cso")));

    const std::vector<D3D11_INPUT_ELEMENT_DESC> ied =
    {
        {"Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    addSharedBind(gfx.createNewGraphicsBoundObject<InputLayout>(ied, pvsbc));

    addSharedBind(gfx.createNewGraphicsBoundObject<Topology>(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST));

    addSharedBind(gfx.createNewGraphicsBoundObject<VertexConstantBuffer<DirectX::XMMATRIX>>(0, 
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixScaling(char_width, char_height, 1.0f) * DirectX::XMMatrixTranslation(left_down_x, left_down_y, 0.0f)
        )
    ));

    unsigned int stringLength = static_cast<int>(text.size());

    std::map<char, int> texturePositionMap;

    std::vector<DirectX::XMFLOAT3> vertices;
    std::vector<DirectX::XMFLOAT2> texturecoords;
    std::vector<unsigned short> indices;

    for (int i = 0; i < stringLength; i++){
        const char& character = text[i];

        if(character < FIRST_USEFUL_ASCII_CHAR || character >= FIRST_USEFUL_ASCII_CHAR+NUM_USEFUL_ASCII_CHARS){
            throw std::exception("StaticScreenTextDrawer received invalid characters");
        }

        int textureIndex;
        std::map<char, int>::iterator it;
        if((it = texturePositionMap.find(character))==texturePositionMap.end()){
            textureIndex = texturePositionMap.size();
            texturePositionMap[character] = textureIndex;
        }
        else{
            textureIndex = it->second;
        }

        vertices.push_back({(float)i, 0.0f, 0.0f});
        vertices.push_back({(float)(i+1), 0.0f, 0.0f});
        vertices.push_back({(float)i, 1.0f, 0.0f});
        vertices.push_back({(float)(i+1), 1.0f, 0.0f});

        texturecoords.push_back({(float)textureIndex, 0.0f});
        texturecoords.push_back({(float)textureIndex+1.0f, 0.0f});
        texturecoords.push_back({(float)textureIndex, 1.0f});
        texturecoords.push_back({(float)textureIndex+1.0f, 1.0f});

        indices.push_back(i*VERTICES_PER_CHARACTER);
        indices.push_back(i*VERTICES_PER_CHARACTER+2);
        indices.push_back(i*VERTICES_PER_CHARACTER+1);
        indices.push_back(i*VERTICES_PER_CHARACTER+1);
        indices.push_back(i*VERTICES_PER_CHARACTER+2);
        indices.push_back(i*VERTICES_PER_CHARACTER+3);
    }

    for(auto it=texturecoords.begin(); it!=texturecoords.end(); it++){
        it->x /= (float)texturePositionMap.size();
    }

    std::unique_ptr<Texture::Color[]> textureBuffer = std::make_unique<Texture::Color[]>(texturePositionMap.size()*CHAR_PIXEL_HEIGHT*CHAR_PIXEL_WIDTH);
    for(auto it=texturePositionMap.begin(); it!=texturePositionMap.end(); it++){
        for(int j=0; j<CHAR_PIXEL_HEIGHT; j++){
            for(int e=0; e<CHAR_PIXEL_WIDTH; e++){
                if((bitmap[(it->first-FIRST_USEFUL_ASCII_CHAR)*CHAR_PIXEL_HEIGHT+j] << e) & 0b10000000){
                    textureBuffer[(CHAR_PIXEL_HEIGHT-1-j)*texturePositionMap.size()*CHAR_PIXEL_WIDTH+it->second*CHAR_PIXEL_WIDTH+e] = {(unsigned char)(blue*255.0f), (unsigned char)(green*255.0f), (unsigned char)(red*255.0f), 255};
                }
                else{
                    textureBuffer[(CHAR_PIXEL_HEIGHT-1-j)*texturePositionMap.size()*CHAR_PIXEL_WIDTH+it->second*CHAR_PIXEL_WIDTH+e] = {(unsigned char)(blue*255.0f), (unsigned char)(green*255.0f), (unsigned char)(red*255.0f), 0};
                }
            }
        }
    }
    addSharedBind(gfx.createNewGraphicsBoundObject<Texture>(std::move(textureBuffer), static_cast<int>(texturePositionMap.size())*CHAR_PIXEL_WIDTH, CHAR_PIXEL_HEIGHT));


    const void* vertexBuffers[] = {(void*)vertices.data(), (void*)texturecoords.data()};
    const size_t vertexSizes[] = {sizeof(DirectX::XMFLOAT3), sizeof(DirectX::XMFLOAT2)};
    const size_t numVertices[] = {VERTICES_PER_CHARACTER*stringLength, VERTICES_PER_CHARACTER*stringLength};
    addSharedBind(gfx.createNewGraphicsBoundObject<ConstantVertexBuffer>(vertexBuffers, vertexSizes, numVertices, 2));
    addSharedBind(gfx.createNewGraphicsBoundObject<IndexBuffer>(indices));

    setIndexCount(stringLength*INDICES_PER_CHARACTER);
}

void StaticScreenTextDrawer::drawStaticScreenText() const{
    bindSharedBinds(typeid(StaticScreenTextDrawer));
    drawIndexed();
}

template LIBRARY_API std::unique_ptr<DynamicTextDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<StaticScreenTextDrawer> GraphicsEngine::createNewGraphicsBoundObject(const std::string& text, float left_down_x, float left_down_y, float char_width, float char_height, float red, float green, float blue);