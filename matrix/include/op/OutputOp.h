//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_OUTPUTOP_H
#define MATRIX_OUTPUTOP_H

#include "Operator.h"

namespace matrix {
    struct OutputParam : public Parameter {
        OutputParam(MatrixType matrixType) : Parameter(matrixType) {

        }

    };


    template <class T, class xpu>
    class OutputOp : public Operator {
    SAME_FUNCTION(Output);
    DISABLE_COPY_AND_ASSIGN(Output);
        INPUT_TAG(DATA);
    };


    template <typename xpu>
    Operator* CreateOp(OutputParam &param, long *size);



    class OutputOpProp : public OperatorProperty {
    public:
        OutputOpProp();
        OutputOpProp(const MatrixType &type);
        ~OutputOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        OutputParam* param;
    };
}

REGISTER_OP_PROPERTY(output, OutputOpProp);

#endif //MATRIX_OUTPUTOP_H
