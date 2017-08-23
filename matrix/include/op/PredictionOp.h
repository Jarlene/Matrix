//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_PREDICTIONOP_H
#define MATRIX_PREDICTIONOP_H


#include "Operator.h"


namespace matrix {


    struct PredictionParam : public Parameter {
        PredictionParam(MatrixType matrixType) : Parameter(matrixType) {}
    };


    template <class T, class xpu>
    class PredictionOp : public Operator {
    SAME_FUNCTION(Prediction);
    DISABLE_COPY_AND_ASSIGN(Prediction);
    };


    template <typename xpu>
    Operator* CreateOp(PredictionParam &param, long *size);



    class PredictionOpProp : public OperatorProperty {
    public:
        PredictionOpProp();
        PredictionOpProp(const MatrixType &type);
        ~PredictionOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape*> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape*> &outShape,
                                         std::map<std::string, Any> &args) ;
        virtual void SwitchType(const MatrixType &type);
    private:
        PredictionParam* param;
    };
}

REGISTER_OP_PROPERTY(prediction, PredictionOpProp);

#endif //MATRIX_PREDICTIONOP_H
