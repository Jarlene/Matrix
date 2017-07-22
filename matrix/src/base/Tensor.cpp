//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/base/Tensor.h"



namespace matrix {

    template <class T, int dimension>
    Tensor<T, dimension>::Tensor() {

    }


}

template class matrix::Tensor<double, 2>;
