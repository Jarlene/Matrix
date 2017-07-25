//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_SHAPE_H
#define MATRIX_SHAPE_H


#include <cassert>
#include <iostream>
#include <vector>

namespace matrix {

    class Shape {
    public:
        Shape(const int* shape);

        Shape(const Shape &shape);

        void reShape(const Shape &shape);

        const size_t size() const ;

        const int operator[](int idx) const ;

    private:
        std::vector<int> shape_;
    };




    template <typename... T>
    inline Shape ShapeN(T... args) {
        const int dims = sizeof...(args);
        int len[dims] = {args...};
        Shape shape(len);
        return shape;
    }
}


#endif //MATRIX_SHAPE_H
