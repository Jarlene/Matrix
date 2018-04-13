//
// Created by Jarlene on 2018/4/13.
//

#ifndef MATRIX_CREATOR_H
#define MATRIX_CREATOR_H

#include <memory>

namespace matrix {

    template <class Construct, typename ... Args>
    std::shared_ptr<Construct> create(Args... args) {
        return std::make_shared<Construct>(args...);
    };



}

#endif //MATRIX_CREATOR_H
