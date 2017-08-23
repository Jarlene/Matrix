//
// Created by  Jarlene on 2017/8/9.
//

#ifndef MATRIX_APPLYGRADOP_H
#define MATRIX_APPLYGRADOP_H


#include "Operator.h"

namespace matrix {

    struct UpdateParam : public Parameter {
        UpdateParam(MatrixType matrixType) : Parameter(matrixType) {
        }

    };

    template <class T, class xpu>
    class UpdateOp : public Operator {
    SAME_FUNCTION(Update);
    DISABLE_COPY_AND_ASSIGN(Update);
        INPUT_TAG(VARIABLE, GRAD_VARIABLE);
    };


    template <typename xpu>
    Operator* CreateOp(UpdateParam &param);


    class UpdateOpProp : public OperatorProperty {
    public:
        UpdateOpProp();
        UpdateOpProp(const MatrixType &type);
        ~UpdateOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape*> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape*> &outShape,
                                         std::map<std::string, Any> &args) ;
        virtual void SwitchType(const MatrixType &type);
    private:
        UpdateParam* param;
    };
}

REGISTER_OP_PROPERTY(applyGrad, UpdateOpProp);

#endif //MATRIX_APPLYGRADOP_H
