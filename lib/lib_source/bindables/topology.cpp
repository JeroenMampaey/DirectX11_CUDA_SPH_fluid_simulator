#include "topology.h"

Topology::Topology(std::shared_ptr<BindableHelper> pHelper, D3D11_PRIMITIVE_TOPOLOGY type) 
	:
	Bindable(std::move(pHelper)),
	type(type)
{}

void Topology::bind() const{
	helper->getContext().IASetPrimitiveTopology(type);
}