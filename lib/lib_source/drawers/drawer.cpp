#include "drawer.h"
#include "../bindables/bindables_includes.h"
#include "../helpers.h"

Drawer::Drawer(std::shared_ptr<DrawerHelper> helper) noexcept
    :
    GraphicsBoundObject(helper)
{}

Drawer::~Drawer() noexcept = default;

void Drawer::addSharedBind(std::unique_ptr<Bindable> bind) noexcept{
    sharedBinds.push_back(std::move(bind));
}

void Drawer::setIndexCount(int indexCount) noexcept{
    this->indexCount = indexCount;
}


void Drawer::setInstanceCount(int instanceCount) noexcept{
    this->instanceCount = instanceCount;
}

void Drawer::setVertexCount(int vertexCount) noexcept{
    this->vertexCount = vertexCount;
}

void Drawer::bindSharedBinds(std::type_index typeIndex) const{
    if(helper->getLastDrawer()!=typeIndex){
        for(auto& bind : sharedBinds){
            bind->bind();
        }
        helper->setLastDrawer(typeIndex);
    }
}

void Drawer::drawIndexed() const{
    if(indexCount==-1){
        throw std::exception("Tried drawing an object with an invalid indexCount");
    }

    helper->drawIndexed(indexCount);
}

void Drawer::drawInstanced() const{
    if(vertexCount==-1){
        throw std::exception("Tried drawing an object with an invalid vertexCount");
    }
    if(instanceCount==-1){
        throw std::exception("Tried drawing an object with an invalid instanceCount");
    }

    helper->drawInstanced(vertexCount, instanceCount);
} 