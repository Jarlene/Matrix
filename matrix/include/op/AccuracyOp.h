//
// Created by Jarlene on 2017/7/27.
//

#ifndef MATRIX_ACCURACYOP_H
#define MATRIX_ACCURACYOP_H

#include "Operator.h"


namespace matrix {

    class AccuracyParam : Parameter {

    };

    template <class T, class Context>
    class AccuracyOp : public Operator {
    SAME_FUNCTION(Accuracy);
    DISABLE_COPY_AND_ASSIGN(Accuracy);
    };




    template <typename Context>
    Operator* CreateOp(AccuracyParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);
}


#endif //MATRIX_ACCURACYOP_H
