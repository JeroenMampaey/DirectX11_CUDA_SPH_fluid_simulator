#pragma once

#include "graphics_engine.h"

class ViewProjectionControllerHelper;

class ViewProjectionController : public GraphicsBoundObject<ViewProjectionControllerHelper>{
    public:
        LIBRARY_API ~ViewProjectionController() noexcept;
        LIBRARY_API void setProjection(DirectX::FXMMATRIX& proj) const;
        LIBRARY_API void setView(DirectX::FXMMATRIX& v) const;
        LIBRARY_API DirectX::XMMATRIX getProjection() const;
	    LIBRARY_API DirectX::XMMATRIX getView() const;

#ifndef READ_FROM_LIB_HEADER
        ViewProjectionController(std::shared_ptr<ViewProjectionControllerHelper> pViewProjectionControllerHelper) noexcept;
#endif
};