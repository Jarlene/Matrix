//
// Created by Jarlene on 2017/11/8.
//

#include "matrix/include/api/ParameterCollection.h"


namespace matrix {

    void ParameterCollection::add(const Symbol &express) {
        this->express.push_back(express.nodePtr);
    }

    void ParameterCollection::add_param(const Symbol &symbol) {
        this->params.push_back(symbol.nodePtr);
    }

    void ParameterCollection::add_input(const Symbol &symbol) {
        this->inputs.push_back(symbol.nodePtr);
    }
}