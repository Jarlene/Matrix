//
// Created by  Jarlene on 2017/8/9.
//

#ifndef MATRIX_APPLYGRADOP_H
#define MATRIX_APPLYGRADOP_H


#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class UpdateOp : public Operator {
    SAME_FUNCTION(Update);
        virtual bool ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) override ;
    DISABLE_COPY_AND_ASSIGN(Update);
        INPUT_TAG(VARIABLE, GRAD_VARIABLE, MOMENTUM);
    private:
        long num_of_pass = 0;

    };




    class UpdateOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(UpdateOpProp)

    };
}

REGISTER_OP_PROPERTY(applyGrad, UpdateOpProp);

#endif //MATRIX_APPLYGRADOP_H
