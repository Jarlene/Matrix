//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_SHAPE_H
#define MATRIX_SHAPE_H


#include <assert.h>
#include <iostream>

namespace matrix {

    template <int dimension>
    class Shape {
    public:
        Shape(const int* shape){
        #pragma unroll
            for (int i = 0; i < dimension; ++i) {
                this->shape_[i] = shape[i];
            }
        }



    private:
        int shape_[dimension];
    };



    template <int dimension, typename... T>
    inline Shape<dimension> ShapeN(T... args) {
        const int dims = sizeof...(args);
        assert(dims == dimension);
        int len[dims] = {args...};
        Shape<dimension> shape(len);
        return shape;
    }
}


#endif //MATRIX_SHAPE_H
