#pragma once

#include <windows.h>
#include "exceptions.h"
#include <d3d11.h>
#include <wrl/client.h>
#include <DirectXMath.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "D3DCompiler.lib")
#include "exports.h"

#define HEIGHT 700
#define WIDTH 1280
#define GRAPHICS_ENGINE_UPDATE_TIMER_ID 101

class LIBRARY_API GraphicsEngine{
    friend class Bindable;

    public:
        GraphicsEngine(HWND hWnd, UINT msPerFrame);
        virtual ~GraphicsEngine();
        void DrawIndexed(UINT count) noexcept;
        void SetProjection(DirectX::FXMMATRIX proj) noexcept;
	    DirectX::XMMATRIX GetProjection() const noexcept;
        std::exception_ptr getThrownException() const noexcept;
    
    protected:
        void ClearBuffer(float red, float green, float blue) noexcept;
        void EndFrame();
        virtual void update() = 0;
        void setThrownException(std::exception_ptr thrownException) noexcept;

        DirectX::XMMATRIX projection;
        Microsoft::WRL::ComPtr<ID3D11Device> pDevice;
        Microsoft::WRL::ComPtr<IDXGISwapChain> pSwap;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> pContext;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> pTarget;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> pDSV;
    
    private:
        static void CALLBACK requestUpdate(HWND hWnd, UINT uMsg, UINT_PTR idEvent, DWORD dwTime) noexcept;
        HWND hWnd;
        std::exception_ptr thrownException = nullptr;
};