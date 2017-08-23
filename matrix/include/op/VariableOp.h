//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_VARIABLEOP_H
#define MATRIX_VARIABLEOP_H

#include "Operator.h"

namespace matrix {


    struct VariableParam : public Parameter {
        VariableParam(MatrixType matrixType) : Parameter(matrixType) {

        }

    };

    template <class T, class xpu>
    class VariableOp : public Operator {
    SAME_FUNCTION(Variable);
    DISABLE_COPY_AND_ASSIGN(Variable);
        INPUT_TAG(INPUT);
        OUTPUT_TAG(OUT);
    };


    template <typename xpu>
    Operator* CreateOp(VariableParam &param, long *size);


    class VariableOpProp : public OperatorProperty {
    public:
        VariableOpProp();
        VariableOpProp(const MatrixType &type);
        ~VariableOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape*> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape*> &outShape,
                                         std::map<std::string, Any> &args) ;
        virtual void SwitchType(const MatrixType &type);
    private:
        VariableParam* param;
    };
}
REGISTER_OP_PROPERTY(variable, VariableOpProp);

#endif //MATRIX_VARIABLEOP_H
