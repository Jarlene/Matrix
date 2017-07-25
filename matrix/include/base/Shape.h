//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_SHAPE_H
#define MATRIX_SHAPE_H


#include <cassert>
#include <iostream>

namespace matrix {

    template <int dimension>
    class Shape {
    public:
        Shape(const int* shape);

        Shape(const Shape<dimension> &shape);

        void reShape(const Shape &shape);

        const size_t size() const ;

        const int operator[](int idx) const ;

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
