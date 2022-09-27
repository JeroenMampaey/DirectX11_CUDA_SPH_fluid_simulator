#pragma once

#include "exceptions.h"
#include "exports.h"
#include <vector>
#include "windows_includes.h"
#include <memory>
#include <unordered_map>
#include <type_traits>


#define RATE_IS_INVALID(rate) ((rate) <= 0.0)

#define HEIGHT 700
#define WIDTH 1280
#define GRAPHICS_ENGINE_UPDATE_TIMER_ID 101
#define NVIDIA_VENDOR_ID 4318
#define AMD_VENDOR_ID 4098
#define INTEL_VENDOR_ID 32902

class LIBRARY_API GraphicsEngine{
        friend class Bindable;
        friend class Drawer;
    public:
        GraphicsEngine& operator=(const GraphicsEngine& copy) = delete;
        GraphicsEngine& operator=(GraphicsEngine&& copy) = delete;
        
        GraphicsEngine(HWND hWnd, UINT syncInterval);
        ~GraphicsEngine() noexcept;
        void setProjection(DirectX::FXMMATRIX proj) noexcept;
	    DirectX::XMMATRIX getProjection() const noexcept;
        void setView(DirectX::FXMMATRIX v) noexcept;
	    DirectX::XMMATRIX getView() const noexcept;
        float getRefreshRate() const noexcept;
        void beginFrame(float red, float green, float blue) noexcept;
        void endFrame() const;

        template<class T, class V, class=std::enable_if_t<std::is_base_of_v<Drawer,T>>>
        std::shared_ptr<T> createNewDrawer(V& args){
            std::shared_ptr<T> newDrawer =  std::shared_ptr<T>(new T(this, drawerUidCounter, args));
            drawersMap.insert({drawerUidCounter, newDrawer});
            drawerUidCounter++;
            return newDrawer;
        };

    private:
        std::shared_ptr<class StaticScreenTextDrawer> screenDrawer;

        DirectX::XMMATRIX projection;
        DirectX::XMMATRIX view;

        Microsoft::WRL::ComPtr<ID3D11Device> pDevice;
        Microsoft::WRL::ComPtr<IDXGISwapChain> pSwap;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> pContext;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> pTarget;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> pDSV;

        std::unordered_map<int, std::weak_ptr<Drawer>> drawersMap;

        UINT syncInterval;
        float refreshRate = -1.0f;
        int lastDrawer = -1;
        int drawerUidCounter = 0;
};