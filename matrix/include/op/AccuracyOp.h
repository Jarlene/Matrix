//
// Created by Jarlene on 2017/7/27.
//

#ifndef MATRIX_ACCURACYOP_H
#define MATRIX_ACCURACYOP_H

#include "Operator.h"


namespace matrix {

    template <class T, class Context>
    class AccuracyOp : public Operator {
    SAME_FUNCTION(Accuracy);
    DISABLE_COPY_AND_ASSIGN(Accuracy);
        INPUT_TAG(PREDICTION, LABEL);
    };


    class AccuracyOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(AccuracyOpProp)
    };
}


REGISTER_OP_PROPERTY(accuracy, AccuracyOpProp);
#endif //MATRIX_ACCURACYOP_H
