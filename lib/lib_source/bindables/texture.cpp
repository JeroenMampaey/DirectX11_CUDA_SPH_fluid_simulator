#include "texture.h"

Texture::Color::Color(unsigned char b, unsigned char g, unsigned char r, unsigned char a) noexcept
    :
	b(b),
	g(g),
	r(r),
	a(a)
{}

Texture::Color::Color() noexcept
    :
	b(0),
	g(0),
	r(0),
	a(0)
{}

Texture::Texture(GraphicsEngine& gfx, std::unique_ptr<Color[]> pBuffer, unsigned int width, unsigned int height){
    HRESULT hr;

	D3D11_TEXTURE2D_DESC textureDesc = {};
	textureDesc.Width = width;
	textureDesc.Height = height;
	textureDesc.MipLevels = 1;
	textureDesc.ArraySize = 1;
	textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.SampleDesc.Quality = 0;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = 0;
	D3D11_SUBRESOURCE_DATA sd = {};
	sd.pSysMem = pBuffer.get();
	sd.SysMemPitch = width*sizeof(Texture::Color);
	Microsoft::WRL::ComPtr<ID3D11Texture2D> pTexture;
    GFX_THROW_FAILED(getDevice(gfx)->CreateTexture2D(&textureDesc, &sd, &pTexture));

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = textureDesc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;
    GFX_THROW_FAILED(getDevice(gfx)->CreateShaderResourceView(pTexture.Get(), &srvDesc, &pTextureView));
}

void Texture::bind(GraphicsEngine& gfx, DrawableState& drawableState){
    getContext(gfx)->PSSetShaderResources(0, 1, pTextureView.GetAddressOf());
}