cbuffer CBuf{
    float4 face_colors;
};

float4 main(float2 tc : TexCoord) : SV_Target{
    float distance = tc.x*tc.x+tc.y*tc.y;
    clip(distance > 1.0f ? -1 : 1);
    return face_colors;
}