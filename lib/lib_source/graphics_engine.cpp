#include "graphics_engine.h"
#include "drawable.h"

GraphicsEngine::GraphicsEngine(HWND hWnd, UINT syncInterval) : syncInterval(syncInterval) {
    if(syncInterval<1 || syncInterval>4){
        throw std::exception("GraphicsEngine constructor syncInterval parameter is invalid, parameter must be between 1 and 4.");
    }

    HRESULT hr;

    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferDesc.Width = 0;
	sd.BufferDesc.Height = 0;
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

    SetWindowTextA(hWnd, ("Vendor id: "+std::to_string(adapterDesc.VendorId)+", DirectX pipeline refreshrate: "+std::to_string(refreshRate)).c_str());

    Microsoft::WRL::ComPtr<ID3D11Resource> pBackbuffer = nullptr;
    GFX_THROW_FAILED(pSwap->GetBuffer(0, __uuidof(ID3D11Resource), &pBackbuffer));
    GFX_THROW_FAILED(pDevice->CreateRenderTargetView(
        pBackbuffer.Get(),
        nullptr,
        &pTarget
    ));

    D3D11_DEPTH_STENCIL_DESC dsDesc = {};
	dsDesc.DepthEnable = TRUE;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsDesc.DepthFunc = D3D11_COMPARISON_LESS;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState> pDSState;
	GFX_THROW_FAILED(pDevice->CreateDepthStencilState(&dsDesc, &pDSState));

	pContext->OMSetDepthStencilState(pDSState.Get(), 1);

    Microsoft::WRL::ComPtr<ID3D11Texture2D> pDepthStensil;
	D3D11_TEXTURE2D_DESC descDepth = {};
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
	descDSV.Format = DXGI_FORMAT_D32_FLOAT;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;
	GFX_THROW_FAILED(pDevice->CreateDepthStencilView(pDepthStensil.Get(), &descDSV, &pDSV));

	pContext->OMSetRenderTargets(1, pTarget.GetAddressOf(), pDSV.Get());

	D3D11_VIEWPORT vp;
	vp.Width = WIDTH;
	vp.Height = HEIGHT;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0.0f;
	vp.TopLeftY = 0.0f;
	pContext->RSSetViewports(1, &vp);
}

void GraphicsEngine::beginFrame(float red, float green, float blue) const noexcept{
    pContext->OMSetRenderTargets(1, pTarget.GetAddressOf(), pDSV.Get());
    const float color[] = {red, green, blue, 1.0};
    pContext->ClearRenderTargetView(pTarget.Get(), color);
    pContext->ClearDepthStencilView(pDSV.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
}

void GraphicsEngine::endFrame() const{
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

void GraphicsEngine::draw(Drawable& drawable) const{
    int indexCount = drawable.getIndexCount();
    if(indexCount==-1) return;

    DrawableState& state = drawable.getState();

    for(auto& b : drawable.getBinds()){
        b->bind(*this, state);
    }

    for(auto& b : drawable.getSharedBinds()){
        b->bind(*this, state);
    }

    pContext->DrawIndexed(indexCount, 0, 0);
}

float GraphicsEngine::getRefreshRate() const noexcept{
    return refreshRate;
}