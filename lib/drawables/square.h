#pragma once

#include "../drawable.h"

class Square : public Drawable<Square>{
    public:
        Square(GraphicsEngine& gfx, float width, float height, float x, float y);
        DirectX::XMMATRIX GetTransformXM() const noexcept;
        void update(float new_x, float new_y) noexcept;
        float getX() const noexcept;
        float getY() const noexcept;
    protected:
        void createSharedBinds(GraphicsEngine& gfx) override;
    private:
        float width;
        float height;
        float x;
        float y;
};