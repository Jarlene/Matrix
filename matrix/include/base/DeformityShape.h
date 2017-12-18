//
// Created by Jarlene on 2017/12/14.
//

#ifndef MATRIX_DEFORMITYSHAPE_H
#define MATRIX_DEFORMITYSHAPE_H

#include "Shape.h"
#include <map>

namespace matrix {

    class DeformityShape : public Shape {
    public:

        DeformityShape() = default;

        void ReShape(const std::vector<Shape*> &shapes);

        void ReShape(const std::vector<Shape> &shapes);

        DeformityShape& operator=(const DeformityShape& other);

        DeformityShape& operator=(const Shape& other);

        const bool operator==(const Shape& shape) const;

        const bool operator==(const DeformityShape& shape) const;

        const Shape operator[](int level) const ;

        void Append(int level, int idx);

        void Append(int idx);

        const size_t Rank() const ;

        const size_t Size() const ;

        const Shape At(int level) const;

        const int At(int level, int idx) const ;

    private:
        std::vector<std::vector<int>> shape_;
    };
}



#endif //MATRIX_DEFORMITYSHAPE_H
