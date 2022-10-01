#include "graphics_engine.h"
#include <sstream>
#include "helpers.h"
#include "drawers/drawers_includes.h"
#include "exceptions.h"

GraphicsEngine::GraphicsEngine(HWND hWnd, UINT syncInterval) 
    : 
    syncInterval(syncInterval)
{
    if(syncInterval<1 || syncInterval>4){
        throw std::exception("GraphicsEngine constructor syncInterval parameter is invalid, parameter must be between 1 and 4.");
    }

    HRESULT hr;

    DXGI_SWAP_CHAIN_DESC sd = {};
    ZeroMemory(&sd, sizeof(DXGI_SWAP_CHAIN_DESC));
    sd.BufferDesc.Width = WIDTH;
	sd.BufferDesc.Height = HEIGHT;
	sd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	sd.BufferDesc.RefreshRate.Numerator = 0;
	sd.BufferDesc.RefreshRate.Denominator = 0;
	sd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	sd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.BufferCount = 2;
	sd.OutputWindow = hWnd;
	sd.Windowed = TRUE;
	sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
	sd.Flags = 0;

    Microsoft::WRL::ComPtr<IDXGIFactory> pFactory = nullptr;
    GFX_THROW_FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory) ,(void**)&pFactory));
    Microsoft::WRL::ComPtr<IDXGIAdapter> pAdapter = nullptr;
    D3D_DRIVER_TYPE driverType = D3D_DRIVER_TYPE_HARDWARE;
    DXGI_ADAPTER_DESC adapterDesc;

    for (UINT i = 0; (hr = pFactory->EnumAdapters(i, &pAdapter)) != DXGI_ERROR_NOT_FOUND; i++) 
    {
        if(FAILED(hr)){
            GFX_THROW_FAILED(hr);
        }

        if(i==0){
            Microsoft::WRL::ComPtr<IDXGIOutput> dxgiOutput;
            if(hr = pAdapter->EnumOutputs(0, &dxgiOutput) != DXGI_ERROR_NOT_FOUND){
                if(FAILED(hr)){
                    GFX_THROW_FAILED(hr);
                }
                Microsoft::WRL::ComPtr<IDXGIOutput1> dxgiOutput1;
                GFX_THROW_FAILED(dxgiOutput.As(&dxgiOutput1));
                DXGI_OUTPUT_DESC outputDes{};
                GFX_THROW_FAILED(dxgiOutput->GetDesc(&outputDes));
                MONITORINFOEXW info;
                info.cbSize = sizeof(info);
                if(GetMonitorInfoW(outputDes.Monitor, &info)==0)
                {
                    CHWND_LAST_EXCEPT();
                }
                UINT32 requiredPaths, requiredModes;
                if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &requiredPaths, &requiredModes) != ERROR_SUCCESS)
                {
                    CHWND_LAST_EXCEPT();
                }
                std::vector<DISPLAYCONFIG_PATH_INFO> paths(requiredPaths);
                std::vector<DISPLAYCONFIG_MODE_INFO> modes2(requiredModes);
                if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &requiredPaths, paths.data(), &requiredModes, modes2.data(), nullptr) != ERROR_SUCCESS)
                {
                    CHWND_LAST_EXCEPT();
                }
                for (auto& p : paths) {
                    DISPLAYCONFIG_SOURCE_DEVICE_NAME sourceName;
                    sourceName.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
                    sourceName.header.size = sizeof(sourceName);
                    sourceName.header.adapterId = p.sourceInfo.adapterId;
                    sourceName.header.id = p.sourceInfo.id;
                    if (DisplayConfigGetDeviceInfo(&sourceName.header) == ERROR_SUCCESS)
                    {
                        if (wcscmp(info.szDevice, sourceName.viewGdiDeviceName) == 0) {
                                UINT numerator = p.targetInfo.refreshRate.Numerator;
                                UINT denominator = p.targetInfo.refreshRate.Denominator;
                                refreshRate = (float)numerator / (float)(syncInterval*denominator);
                                break;
                        }
                    }
                }
            }
        }

        GFX_THROW_FAILED(pAdapter->GetDesc(&adapterDesc));
        if(adapterDesc.VendorId==NVIDIA_VENDOR_ID){
            driverType = D3D_DRIVER_TYPE_UNKNOWN;
            break;
        }
    }

    GFX_THROW_FAILED(D3D11CreateDeviceAndSwapChain(
        pAdapter.Get(),
        driverType,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &sd,
        &pSwap,
        &pDevice,
        nullptr,
        &pContext
    ));

    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice = nullptr;
    GFX_THROW_FAILED(pDevice->QueryInterface(__uuidof(IDXGIDevice), &dxgiDevice));
    GFX_THROW_FAILED(dxgiDevice->GetAdapter(&pAdapter));
    GFX_THROW_FAILED(pAdapter->GetDesc(&adapterDesc));

    std::ostringstream specifications;
    specifications.precision(6);
    switch(adapterDesc.VendorId){
        case NVIDIA_VENDOR_ID:
            specifications << "NVIDIA";
            break;
        case AMD_VENDOR_ID:
            specifications << "AMD";
            break;
        case INTEL_VENDOR_ID:
            specifications << "INTEL";
            break;
        default:
            specifications << "UNKNOWN";
    }
    specifications << ", " << refreshRate << " Hz";
    std::string specString = specifications.str();
    
    Microsoft::WRL::ComPtr<ID3D11Resource> pBackbuffer = nullptr;
    GFX_THROW_FAILED(pSwap->GetBuffer(0, __uuidof(ID3D11Resource), &pBackbuffer));
    GFX_THROW_FAILED(pDevice->CreateRenderTargetView(
        pBackbuffer.Get(),
        nullptr,
        &pTarget
    ));

    D3D11_DEPTH_STENCIL_DESC dsDesc = {};
    ZeroMemory(&dsDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	dsDesc.DepthEnable = TRUE;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsDesc.DepthFunc = D3D11_COMPARISON_LESS;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState> pDSState;
	GFX_THROW_FAILED(pDevice->CreateDepthStencilState(&dsDesc, &pDSState));

	pContext->OMSetDepthStencilState(pDSState.Get(), 1);

    Microsoft::WRL::ComPtr<ID3D11Texture2D> pDepthStensil;
	D3D11_TEXTURE2D_DESC descDepth = {};
    ZeroMemory(&descDepth, sizeof(D3D11_TEXTURE2D_DESC));
	descDepth.Width = WIDTH;
	descDepth.Height = HEIGHT;
	descDepth.MipLevels = 1;
	descDepth.ArraySize = 1;
	descDepth.Format = DXGI_FORMAT_D32_FLOAT;
	descDepth.SampleDesc.Count = 1;
	descDepth.SampleDesc.Quality = 0;
	descDepth.Usage = D3D11_USAGE_DEFAULT;
	descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	GFX_THROW_FAILED(pDevice->CreateTexture2D(&descDepth, nullptr, &pDepthStensil));

	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV = {};
    ZeroMemory(&descDSV, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
	descDSV.Format = DXGI_FORMAT_D32_FLOAT;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;
	GFX_THROW_FAILED(pDevice->CreateDepthStencilView(pDepthStensil.Get(), &descDSV, &pDSV));

	pContext->OMSetRenderTargets(1, pTarget.GetAddressOf(), pDSV.Get());

	D3D11_VIEWPORT vp;
    ZeroMemory(&vp, sizeof(D3D11_VIEWPORT));
	vp.Width = WIDTH;
	vp.Height = HEIGHT;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0.0f;
	vp.TopLeftY = 0.0f;
	pContext->RSSetViewports(1, &vp);

    projection = DirectX::XMMatrixIdentity();
    view = DirectX::XMMatrixIdentity();

    screenDrawer = createNewGraphicsBoundObject<StaticScreenTextDrawer>(specString, -1.0f, 0.9f, 0.05f, 0.1f, 1.0f, 1.0f, 1.0f);
}

void GraphicsEngine::beginFrame(float red, float green, float blue) noexcept{
    pContext->OMSetRenderTargets(1, pTarget.GetAddressOf(), pDSV.Get());
    const float color[] = {red, green, blue, 1.0f};
    pContext->ClearRenderTargetView(pTarget.Get(), color);
    pContext->ClearDepthStencilView(pDSV.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    lastDrawer = typeid(InvalidDrawer);
}

void GraphicsEngine::endFrame() const{
    screenDrawer->drawStaticScreenText();
    HRESULT hr;
    if(FAILED(hr = pSwap->Present(syncInterval, 0))){
        if(hr == DXGI_ERROR_DEVICE_REMOVED){
            throw GFX_DEVICE_REMOVED_EXCEPT(pDevice->GetDeviceRemovedReason());
        }
        else{
            GFX_THROW_FAILED(hr);
        }
    }
}

void GraphicsEngine::setProjection(DirectX::FXMMATRIX proj) noexcept{
	projection = proj;
}

DirectX::XMMATRIX GraphicsEngine::getProjection() const noexcept{
	return projection;
}

void GraphicsEngine::setView(DirectX::FXMMATRIX v) noexcept{
    view = v;
}

DirectX::XMMATRIX GraphicsEngine::getView() const noexcept{
    return view;
}

float GraphicsEngine::getRefreshRate() const noexcept{
    return refreshRate;
}

GraphicsEngine::~GraphicsEngine() noexcept{
    for(const std::pair<std::type_index, std::shared_ptr<Helper>>& pair : helpersMap){
        pair.second->pGfx = nullptr;
    }
}

template<class T, class... Args>
std::unique_ptr<T> GraphicsEngine::createNewGraphicsBoundObject(Args... args){
    std::type_index typeIndex = typeid(typename T::HelperType);
    std::shared_ptr<typename T::HelperType> pHelper;
    auto it = helpersMap.find(typeIndex);
    if(it !=helpersMap.end()){
        pHelper = std::static_pointer_cast<typename T::HelperType>(it->second);
    }
    else{
        pHelper = std::shared_ptr<typename T::HelperType>(new typename T::HelperType(this));
        helpersMap.insert({typeIndex, pHelper});
    }
    return std::unique_ptr<T>(new T(pHelper, std::forward<Args>(args)...));
}

template LIBRARY_API std::unique_ptr<FilledCircleDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<FilledCircleInstanceDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<FilledRectangleDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<HollowRectangleDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<LineDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<DynamicTextDrawer> GraphicsEngine::createNewGraphicsBoundObject(float red, float green, float blue);
template LIBRARY_API std::unique_ptr<StaticScreenTextDrawer> GraphicsEngine::createNewGraphicsBoundObject(const std::string& text, float left_down_x, float left_down_y, float char_width, float char_height, float red, float green, float blue);
template LIBRARY_API std::unique_ptr<CpuAccessibleFilledCircleInstanceBuffer> GraphicsEngine::createNewGraphicsBoundObject(int numberOfCircles, float radius);
