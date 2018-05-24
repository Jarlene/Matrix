//
// Created by Jarlene on 2017/12/14.
//

#ifndef MATRIX_DEFORMITYSHAPE_H
#define MATRIX_DEFORMITYSHAPE_H

#include "Shape.h"
#include <map>

namespace matrix {

    class LoShape : public Shape {
    public:

        LoShape() = default;

        explicit LoShape(const Shape &shape);

        void ReShape(const std::vector<Shape*> &shapes);

        void ReShape(const std::vector<Shape> &shapes);

        LoShape& operator=(const LoShape& other);

        LoShape& operator=(const Shape& other);

        const bool operator==(const Shape& shape) const;

        const bool operator==(const LoShape& shape) const;

        const Shape operator[](int level) const ;

        const int operator()(int level, int idx) const ;

        void Append(int level, int idx);

        void Append(int idx);

        const size_t Rank() const ;

        const size_t Size() const ;

        const Shape At(int level) const;

        const int At(int level, int idx) const ;

    private:
        std::vector<Shape> shape_;
    };


    template <typename... T>
    inline LoShape ShapeN(T... args) {
        LoShape shape;
        shape.ReShape({args...});
        return shape;
    }
}



#endif //MATRIX_DEFORMITYSHAPE_H
