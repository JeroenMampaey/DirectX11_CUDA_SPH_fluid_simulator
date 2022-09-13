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

class LIBRARY_API GraphicsEngine{
    friend class Bindable;

    public:
        GraphicsEngine(HWND hWnd, UINT syncInterval);
        virtual ~GraphicsEngine() = default;
        void drawIndexed(UINT count) noexcept;
        void setProjection(DirectX::FXMMATRIX proj) noexcept;
	    DirectX::XMMATRIX getProjection() const noexcept;
        virtual void update() = 0;
        virtual void mouseMoveEvent(int x, int y) noexcept;
        virtual void mouseLeftClickEvent(int x, int y) noexcept;
        virtual void mouseRightClickEvent(int x, int y) noexcept;
        virtual void keyEvent(WPARAM charCode) noexcept;
    
    protected:
        void beginFrame(float red, float green, float blue) noexcept;
        void endFrame();
        float refreshRate = -1.0f;

        DirectX::XMMATRIX projection;
        Microsoft::WRL::ComPtr<ID3D11Device> pDevice;
        Microsoft::WRL::ComPtr<IDXGISwapChain> pSwap;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> pContext;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> pTarget;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> pDSV;
    
    private:
        UINT syncInterval;
};