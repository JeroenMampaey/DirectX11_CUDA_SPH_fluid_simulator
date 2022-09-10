#include "topology.h"

Topology::Topology(GraphicsEngine& gfx, D3D11_PRIMITIVE_TOPOLOGY type) : type( type )
{}

void Topology::bind(GraphicsEngine& gfx, DrawableState& drawableState){
	getContext(gfx)->IASetPrimitiveTopology(type);
}