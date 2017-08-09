//
// Created by  Jarlene on 2017/8/9.
//

#ifndef MATRIX_APPLYGRADOP_H
#define MATRIX_APPLYGRADOP_H


#include "Operator.h"

namespace matrix {

    struct ApplyGradParam : public Parameter {
        ApplyGradParam(MatrixType matrixType) : Parameter(matrixType) {
        }

    };

    template <class T, class xpu>
    class ApplyGradOp : public Operator {
    SAME_FUNCTION(ApplyGrad);
    DISABLE_COPY_AND_ASSIGN(ApplyGrad);
        INPUT_TAG(VARIABLE, GRAD_VARIABLE);
    };


    template <typename xpu>
    Operator* CreateOp(ApplyGradParam &param);


    class ApplyGradProp : public OperatorProperty {
    public:
        ApplyGradProp();
        ApplyGradProp(const MatrixType &type);
        ~ApplyGradProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                         std::map<std::string, Any> &args) ;
    private:
        ApplyGradParam* param;
    };
}

REGISTER_OP_PROPERTY(applyGrad, ApplyGradProp);

#endif //MATRIX_APPLYGRADOP_H
