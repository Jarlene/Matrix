//
// Created by Jarlene on 2017/11/9.
//

#ifndef MATRIX_RNNGRADOP_H
#define MATRIX_RNNGRADOP_H


#include "Operator.h"

namespace matrix {


    
    template <class T, class xpu>
    class RNNGradOp : public Operator {
    SAME_FUNCTION(RNNGrad);
        virtual bool VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) override ;
        virtual bool ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) override ;
    DISABLE_COPY_AND_ASSIGN(RNNGrad);
    };




    class RNNGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(RNNGradOpProp)
    };
}


REGISTER_OP_PROPERTY(rnn_grad, RNNGradOpProp);




#endif //MATRIX_RNNGRADOP_H
