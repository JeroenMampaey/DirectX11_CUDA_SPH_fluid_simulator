#include "topology.h"

Topology::Topology(std::shared_ptr<BindableHelper> helper, D3D11_PRIMITIVE_TOPOLOGY type) 
	:
	Bindable(helper),
	type(type)
{}

void Topology::bind(){
	helper->getContext().IASetPrimitiveTopology(type);
}