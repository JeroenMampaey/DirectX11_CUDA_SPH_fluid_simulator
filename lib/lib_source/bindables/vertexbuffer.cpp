#include "vertexbuffer.h"

void VertexBuffer::bind(const GraphicsEngine& gfx){
	const UINT offset = 0u;
	getContext(gfx)->IASetVertexBuffers(0, 1, pVertexBuffer.GetAddressOf(), &stride, &offset );
}