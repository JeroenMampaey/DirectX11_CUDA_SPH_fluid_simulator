#include "topology.h"

Topology::Topology(GraphicsEngine& gfx, D3D11_PRIMITIVE_TOPOLOGY type) : type( type )
{}

void Topology::Bind(GraphicsEngine& gfx){
	GetContext(gfx)->IASetPrimitiveTopology(type);
}