//
// Created by  Jarlene on 2017/8/9.
//

#ifndef MATRIX_PRODUCESHAPE_H
#define MATRIX_PRODUCESHAPE_H

#include <cassert>
#include "matrix/include/base/Shape.h"
#include "Logger.h"

namespace matrix {

    inline void static ProduceMulOpShape(std::vector<Shape*> &input, Shape *out) {


        assert(input.size() >= 2);
        auto in1 = input[0];
        auto in2 = input[1];

        if (in1->isConstant() && in2->isConstant()) {
            in1->reShape(ShapeN(1, 1));
            in2->reShape(ShapeN(1, 1));
        } else if (in1->isVector() && in2->isMatrix()) {
            if (in1->At(0) == in2->At(0)) {
                in1->reShape(ShapeN(1, in1->At(0)));
            } else if (in2->At(0) == 1) {
                in1->reShape(ShapeN(in1->At(0), 1));
            }
        } else if (in1->isMatrix() && in2->isVector()) {
            if (in1->At(1) == in2->At(0)) {
                in2->reShape(ShapeN(in2->At(0), 1));
            } else if (in1->At(1) == 1) {
                in2->reShape(ShapeN(1, in2->At(0)));
            }
        } else if (in1->isMatrix() && in2->isMatrix()) {
            assert(in1->At(1) == in2->At(0));
        } else {
            Logger::Global()->Fatal("ProduceMulOpShape cant not support Mul Shape Size\n");
        }
        input[0]->reShape(*in1);
        input[1]->reShape(*in2);
        out->reShape(ShapeN(in1->At(0), in2->At(0)));

    }

}

#endif //MATRIX_PRODUCESHAPE_H
