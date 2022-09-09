#include "vertexbuffer.h"

void VertexBuffer::Bind(GraphicsEngine& gfx, DrawableState& drawableState){
	const UINT offset = 0u;
	GetContext(gfx)->IASetVertexBuffers(0, 1, pVertexBuffer.GetAddressOf(), &stride, &offset );
}