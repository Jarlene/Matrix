//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_FULLCONNECTEDOP_H
#define MATRIX_FULLCONNECTEDOP_H

#include "Operator.h"

namespace matrix {


    struct  FullConnectedParam : public Parameter {
        FullConnectedParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class FullConnectedOp : public Operator {
    SAME_FUNCTION(FullConnected);
        virtual void VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) override;
    DISABLE_COPY_AND_ASSIGN(FullConnected);
        INPUT_TAG(DATA, WEIGHT, BIAS);
    };

    template <typename Context>
    Operator* CreateOp(FullConnectedParam &param, long *size);


    class FullConnectedOpProp : public OperatorProperty {
    public:
        FullConnectedOpProp();
        FullConnectedOpProp(const MatrixType &type);
        ~FullConnectedOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape  *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        FullConnectedParam* param;
    };
}

REGISTER_OP_PROPERTY(fullConnected, FullConnectedOpProp);

#endif //MATRIX_FULLCONNECTEDOP_H
