//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_PREDICTIONOP_H
#define MATRIX_PREDICTIONOP_H


#include "Operator.h"


namespace matrix {


    template <class T, class xpu>
    class PredictionOp : public Operator {
    SAME_FUNCTION(Prediction);
    DISABLE_COPY_AND_ASSIGN(Prediction);
    };





    class PredictionOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(PredictionOpProp)
    };
}

REGISTER_OP_PROPERTY(prediction, PredictionOpProp);

#endif //MATRIX_PREDICTIONOP_H
