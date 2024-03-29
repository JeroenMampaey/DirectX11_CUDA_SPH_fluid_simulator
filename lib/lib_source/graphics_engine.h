#pragma once

#include "win_exception.h"
#include "exports.h"
#include <vector>
#include "windows_includes.h"
#include <memory>
#include <unordered_map>
#include <type_traits>
#include <typeindex>


#define RATE_IS_INVALID(rate) ((rate) <= 0.0)

#define HEIGHT 700
#define WIDTH 1280
#define GRAPHICS_ENGINE_UPDATE_TIMER_ID 101
#define NVIDIA_VENDOR_ID 4318
#define AMD_VENDOR_ID 4098
#define INTEL_VENDOR_ID 32902

template<class T>
class GraphicsBoundObject{
    public:
        typedef T HelperType;
        virtual ~GraphicsBoundObject() noexcept = default;
    
    protected:
        GraphicsBoundObject(std::shared_ptr<T> helper) noexcept : helper(std::move(helper)) {};
        std::shared_ptr<T> helper;
};

class GraphicsEngine{
        friend class Helper;
        friend class DrawerHelper;
        friend class BindableHelper;
        friend class FrameControllerHelper;
        friend class ViewProjectionControllerHelper;
        friend class Window;
    public:
        LIBRARY_API GraphicsEngine& operator=(const GraphicsEngine& copy) = delete;
        LIBRARY_API GraphicsEngine& operator=(GraphicsEngine&& copy) = delete;
        
        LIBRARY_API ~GraphicsEngine() noexcept;
	    LIBRARY_API DirectX::XMMATRIX getProjection() const noexcept;
	    LIBRARY_API DirectX::XMMATRIX getView() const noexcept;
        LIBRARY_API float getRefreshRate() const noexcept;

#ifndef READ_FROM_LIB_HEADER
    template<class T, class... Args>
    std::unique_ptr<T> createNewGraphicsBoundObject(Args... args){
        std::type_index typeIndex = typeid(typename T::HelperType);
        std::shared_ptr<typename T::HelperType> pHelper;
        auto it = helpersMap.find(typeIndex);
        if(it !=helpersMap.end()){
            pHelper = std::static_pointer_cast<typename T::HelperType>(it->second);
        }
        else{
            pHelper = std::make_shared<typename T::HelperType>(typename T::HelperType(this));
            helpersMap.insert({typeIndex, pHelper});
        }
        return std::make_unique<T>(std::move(pHelper), std::forward<Args>(args)...);
    }
#else
    template<class T, class... Args>
    LIBRARY_API std::unique_ptr<T> createNewGraphicsBoundObject(Args... args);
#endif

    private:
        GraphicsEngine(HWND hWnd, UINT syncInterval);

        std::unordered_map<std::type_index, std::shared_ptr<Helper>> helpersMap;

        std::unique_ptr<class StaticScreenTextDrawer> screenDrawer = nullptr;

        DirectX::XMMATRIX projection;
        DirectX::XMMATRIX view;

        Microsoft::WRL::ComPtr<ID3D11Device> pDevice;
        Microsoft::WRL::ComPtr<IDXGISwapChain> pSwap;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> pContext;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> pTarget;
        Microsoft::WRL::ComPtr<ID3D11DepthStencilView> pDSV;

        UINT syncInterval;
        float refreshRate = -1.0f;
        int lastDrawer = -1;
};