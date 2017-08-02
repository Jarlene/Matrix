//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_CONVOLUTIONOP_H
#define MATRIX_CONVOLUTIONOP_H

#include "Operator.h"

namespace matrix {

    class ConvolutionParam : Parameter {

    };

    template <class T, class Context>
    class ConvolutionOp : public Operator {
    public:
        explicit ConvolutionOp(ConvolutionParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~ConvolutionOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(ConvolutionOp);
    };

    template <typename Context>
    Operator* CreateOp(ConvolutionParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_CONVOLUTIONOP_H
