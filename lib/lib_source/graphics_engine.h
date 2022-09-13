#pragma once

#include <windows.h>
#include "exceptions.h"
#include <d3d11.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include "exports.h"
#include <dxgi1_2.h>
#include <d3d10.h>
#include <d3d9.h>
#include <vector>
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3DCompiler.lib")
#pragma comment(lib, "dxgi")

#define RATE_IS_INVALID(rate) (rate <= 0.0)

#define HEIGHT 700
#define WIDTH 1280
#define GRAPHICS_ENGINE_UPDATE_TIMER_ID 101
#define NVIDIA_VENDOR_ID 4318

class Drawable;

class LIBRARY_API GraphicsEngine{
        friend class Bindable;
    public:
        GraphicsEngine(HWND hWnd, UINT syncInterval);
        void setProjection(DirectX::FXMMATRIX proj) noexcept;
	    DirectX::XMMATRIX getProjection() const noexcept;
        void beginFrame(float red, float green, float blue) const noexcept;
        void endFrame() const;
        void draw(Drawable& drawable) const;
        float getRefreshRate() const noexcept;

    private:
        DirectX::XMMATRIX projection;
        Microsoft::WRL::ComPtr<ID3D11Device> pDevice;
        Microsoft::WRL::ComPtr<IDXGISwapChain> pSwap;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> pContext;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> pTarget;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> pDSV;
        UINT syncInterval;
        float refreshRate = -1.0f;
};