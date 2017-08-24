//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_SUBOP_H
#define MATRIX_SUBOP_H

#include "Operator.h"

namespace matrix {

    struct SubParam : public Parameter {
        SubParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class SubOp : public Operator {
    SAME_FUNCTION(Sub);
    DISABLE_COPY_AND_ASSIGN(Sub);
    };

    template <typename xpu>
    Operator* CreateOp(SubParam &param, long *size);


    class SubOpProp : public OperatorProperty {
    public:
        SubOpProp();
        SubOpProp(const MatrixType &type);
        ~SubOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, std::vector<Shape*> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, std::vector<Blob*> &output,
                                         std::vector<Shape*> &inShape, std::vector<Shape*> &outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        SubParam* param;
    };
}

REGISTER_OP_PROPERTY(sub, SubOpProp);

#endif //MATRIX_SUBOP_H
