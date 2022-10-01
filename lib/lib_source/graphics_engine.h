#pragma once

#include "win_exception.h"
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

class GraphicsEngine{
        friend class Bindable;
        friend class DrawerHelper;
        friend class Window;
    public:
        LIBRARY_API GraphicsEngine& operator=(const GraphicsEngine& copy) = delete;
        LIBRARY_API GraphicsEngine& operator=(GraphicsEngine&& copy) = delete;
        
        LIBRARY_API ~GraphicsEngine() noexcept;
        LIBRARY_API void setProjection(DirectX::FXMMATRIX proj) noexcept;
	    LIBRARY_API DirectX::XMMATRIX getProjection() const noexcept;
        LIBRARY_API void setView(DirectX::FXMMATRIX v) noexcept;
	    LIBRARY_API DirectX::XMMATRIX getView() const noexcept;
        LIBRARY_API float getRefreshRate() const noexcept;
        LIBRARY_API void beginFrame(float red, float green, float blue) noexcept;
        LIBRARY_API void endFrame() const;

        template<class T, class... Args>
        LIBRARY_API std::unique_ptr<T> createNewDrawer(Args... args);

    private:
        GraphicsEngine(HWND hWnd, UINT syncInterval);

        std::unique_ptr<class StaticScreenTextDrawer> screenDrawer;

        DirectX::XMMATRIX projection;
        DirectX::XMMATRIX view;

        Microsoft::WRL::ComPtr<ID3D11Device> pDevice;
        Microsoft::WRL::ComPtr<IDXGISwapChain> pSwap;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> pContext;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> pTarget;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> pDSV;

        std::unordered_map<int, std::weak_ptr<DrawerHelper>> drawerHelpersMap;

        UINT syncInterval;
        float refreshRate = -1.0f;
        int lastDrawer = -1;
        int drawerUidCounter = 0;
};