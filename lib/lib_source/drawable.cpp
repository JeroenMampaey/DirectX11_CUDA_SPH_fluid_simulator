#include "drawable.h"

const std::vector<std::unique_ptr<Bindable>>& Drawable::getBinds() const noexcept{
    return binds;
}


int Drawable::getIndexCount() const noexcept{
    return indexCount;
}

void Drawable::addUniqueBind(std::unique_ptr<Bindable> bind) noexcept{
    binds.push_back(std::move(bind));
}

void Drawable::setIndexCount(int indexCount) noexcept{
    this->indexCount = indexCount;
}