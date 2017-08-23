//
// Created by Jarlene on 2017/7/27.
//

#ifndef MATRIX_ACCURACYOP_H
#define MATRIX_ACCURACYOP_H

#include "Operator.h"


namespace matrix {

    struct AccuracyParam : public Parameter {
        AccuracyParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class AccuracyOp : public Operator {
    SAME_FUNCTION(Accuracy);
    DISABLE_COPY_AND_ASSIGN(Accuracy);
        INPUT_TAG(DATA, LABEL);
        OUTPUT_TAG(OUT);
    };




    template <typename xpu>
    Operator* CreateOp(AccuracyParam &param, long *size);


    class AccuracyOpProp : public OperatorProperty {
    public:
        AccuracyOpProp();
        AccuracyOpProp(const MatrixType &type);
        virtual void SwitchType(const MatrixType &type);
        ~AccuracyOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                         std::map<std::string, Any> &args) ;
    private:
        AccuracyParam* param;
    };
}


REGISTER_OP_PROPERTY(accuracy, AccuracyOpProp);
#endif //MATRIX_ACCURACYOP_H
