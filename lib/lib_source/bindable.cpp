#include "bindable.h"

ID3D11DeviceContext* Bindable::getContext(const GraphicsEngine& gfx) noexcept{
    return gfx.pContext.Get();
}

ID3D11Device* Bindable::getDevice(const GraphicsEngine& gfx) noexcept{
    return gfx.pDevice.Get();
}
