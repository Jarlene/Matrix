//
// Created by Jarlene on 2017/11/27.
//

#include "matrix/include/op/BatchNormalizationOp.h"
#include "matrix/include/op/ReduceOp.h"

namespace matrix {

    template <class T, class xpu>
    BatchNormalizationOp<T, xpu>::BatchNormalizationOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool BatchNormalizationOp<T, xpu>::Run() {
        const T* input = Input<T>(DATA);
        const T* gamma = Input<T>(GAMMA);
        const T* beta = Input<T>(BETA);
        T* mean_ptr = Input<T>(MEAN);
        T* var_ptr = Input<T>(VAR);

        auto in = Tensor<T>(input, *inputShapes->at(DATA));
        auto mean = Tensor<T>(mean_ptr, *inputShapes->at(MEAN));
        auto var = Tensor<T>(var_ptr, *inputShapes->at(VAR));
        // 1、input data mean and var;
        Mean(in, 1, mean);
        Var(in, mean, 1, var);
        // 2、for standard input data;
        int batch = inputShapes->at(DATA)->At(0);
        for(int i = 0; i < batch; ++i) {

        }


        return true;
    }

    template <class T, class xpu>
    void BatchNormalizationOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool BatchNormalizationOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    BatchNormalizationOp<T, xpu>::~BatchNormalizationOp() {

    }

    template<class T, class xpu>
    bool BatchNormalizationOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        if (InputSize() == 1) {
            auto in = inputShapes->at(DATA);
            Shape shape;
            for(int i = 1; i < in->Rank(); ++i) {
                shape.Append(in->At(i));
            }
            func({&shape, &shape});
            return true;
        }
        return false;
    }

    void BatchNormalizationOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape.at(0));
    }

    INIT_OPERATOR_PROPERTY_CREATE(BatchNormalizationOpProp, BatchNormalizationOp, true);

}