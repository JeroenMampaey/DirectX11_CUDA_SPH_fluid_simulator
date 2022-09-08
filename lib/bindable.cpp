#include "bindable.h"

ID3D11DeviceContext* Bindable::GetContext(GraphicsEngine& gfx) noexcept{
    return gfx.pContext.Get();
}

ID3D11Device* Bindable::GetDevice(GraphicsEngine& gfx) noexcept{
    return gfx.pDevice.Get();
}
