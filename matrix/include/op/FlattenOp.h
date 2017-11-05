//
// Created by Jarlene on 2017/11/5.
//

#ifndef MATRIX_FLATTENOP_H
#define MATRIX_FLATTENOP_H

#include "Operator.h"

namespace matrix {


    struct FlattenParam : public Parameter{
        FlattenParam(MatrixType matrixType): Parameter(matrixType) {

        };
    };

    template <class T, class Context>
    class FlattenOp : public Operator {
    SAME_FUNCTION(Flatten);
    DISABLE_COPY_AND_ASSIGN(Flatten);
    };

    template <typename xpu>
    Operator* CreateOp(FlattenParam &param, long *size);


    class FlattenOpProp : public OperatorProperty {
    public:
        FlattenOpProp();
        FlattenOpProp(const MatrixType &type);
        ~FlattenOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape* outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape* outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        FlattenParam* param;
    };
}

REGISTER_OP_PROPERTY(flatten, FlattenOpProp);



#endif //MATRIX_FLATTENOP_H
