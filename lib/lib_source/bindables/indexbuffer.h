#pragma once

#include <vector>
#include "bindable.h"

class IndexBuffer : public Bindable{
    public:
        IndexBuffer(std::shared_ptr<BindableHelper> pHelper, const std::vector<unsigned short>& indices);
        void bind() const override;
    protected:
        Microsoft::WRL::ComPtr<ID3D11Buffer> pIndexBuffer;
};