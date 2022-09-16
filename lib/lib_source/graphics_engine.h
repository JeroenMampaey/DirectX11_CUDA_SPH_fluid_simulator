#pragma once

#include "exceptions.h"
#include "exports.h"
#include <vector>
#include "windows_includes.h"
#include <memory>
#include <map>
#include "drawable_manager_base.h"

#define RATE_IS_INVALID(rate) (rate <= 0.0)

#define HEIGHT 700
#define WIDTH 1280
#define GRAPHICS_ENGINE_UPDATE_TIMER_ID 101
#define NVIDIA_VENDOR_ID 4318
#define AMD_VENDOR_ID 4098
#define INTEL_VENDOR_ID 32902

enum DrawableType;

class LIBRARY_API GraphicsEngine{
        friend class Bindable;
    public:
        GraphicsEngine& operator=(const GraphicsEngine& copy) = delete;
        GraphicsEngine& operator=(GraphicsEngine&& copy) = delete;
        
        GraphicsEngine(HWND hWnd, UINT syncInterval);
        void setProjection(DirectX::FXMMATRIX proj) noexcept;
	    DirectX::XMMATRIX getProjection() const noexcept;
        void setView(DirectX::FXMMATRIX v) noexcept;
	    DirectX::XMMATRIX getView() const noexcept;
        float getRefreshRate() const noexcept;
        void drawIndexed(int indexCount) const noexcept;
        Drawable* createDrawable(DrawableType type, DrawableInitializerDesc& desc);
        int removeDrawable(DrawableType type, Drawable* drawable);
        void updateFrame(float red, float green, float blue) const;

    private:
        void beginFrame(float red, float green, float blue) const noexcept;
        void endFrame() const;

        DirectX::XMMATRIX projection;
        DirectX::XMMATRIX view;

        Microsoft::WRL::ComPtr<ID3D11Device> pDevice;
        Microsoft::WRL::ComPtr<IDXGISwapChain> pSwap;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> pContext;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> pTarget;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> pDSV;

        std::map<DrawableType, std::unique_ptr<DrawableManagerBase>> managers;
        UINT syncInterval;
        float refreshRate = -1.0f;
};