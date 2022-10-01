cbuffer CBuf{
	matrix transform;
};

struct VSOut{
	float2 tex : TexCoord;
	float4 pos : SV_Position;
};

VSOut main(float3 pos : Position, float2 tex : TexCoord, float3 instPos : InstancePos){
	VSOut vso;
    pos.x += instPos.x;
    pos.y += instPos.y;
    pos.z += instPos.z;
	vso.pos = mul( float4(pos,1.0f),transform );
	vso.tex = tex;
	return vso;
}