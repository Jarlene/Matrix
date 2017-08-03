//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_ADDOP_H
#define MATRIX_ADDOP_H

#include "Operator.h"

namespace matrix {

    class AddParam : Parameter {
    public:
        std::vector<Shape> inShape;
        std::vector<Blob> in;
        Blob out;
    };

    template <class T, class Context>
    class AddOp : public Operator {
    SAME_FUNCTION(Add);
    DISABLE_COPY_AND_ASSIGN(Add);
        INPUT_TAG(INPUT1, INPUT2);
    private:
        Context context;
        std::vector<Shape> inShape;
        std::vector<Blob> in;
        Blob out;
    };


    template <typename Context>
    Operator* CreateOp(AddParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);



}

#endif //MATRIX_ADDOP_H
