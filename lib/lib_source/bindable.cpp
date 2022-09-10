#include "bindable.h"

ID3D11DeviceContext* Bindable::getContext(GraphicsEngine& gfx) noexcept{
    return gfx.pContext.Get();
}

ID3D11Device* Bindable::getDevice(GraphicsEngine& gfx) noexcept{
    return gfx.pDevice.Get();
}
