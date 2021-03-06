//
// Created by Jarlene on 2017/7/31.
//

#include "matrix/include/op/AccuracyOp.h"


namespace matrix {


    template <class T, class xpu>
    AccuracyOp<T, xpu>::AccuracyOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool AccuracyOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    bool AccuracyOp<T, xpu>::Run() {
        int N = inputShapes->at(PREDICTION)->At(0);
        int D = inputShapes->at(PREDICTION)->At(1);
        int correct = 0;
        for (int i = 0; i < N; ++i) {
            auto label = static_cast<int>(Input<T>(LABEL)[i]);
            int index = 0;
            T max = Input<T>(PREDICTION)[i * D];
            for (int j = 1; j < D; ++j) {
                T pred = Input<T>(PREDICTION)[i * D + j];
                if (pred > max) {
                    index = j;
                    max = pred;
                }
            }
            if (label == index) {
                ++correct;
            }
        }
        assert(correct <= N);
        Output<T>()[0] = correct * T(1.0) / N;
        return true;
    }

    template <class T, class xpu>
    void AccuracyOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu){
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    AccuracyOp<T, xpu>::~AccuracyOp() {

    }



    void AccuracyOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (outShape == nullptr) {
            Logger::Global()->Fatal("AccuracyOp  must has output shape. \n");
        }
        outShape->reShape(ShapeN(1));
    }
    INIT_OPERATOR_PROPERTY_CREATE(AccuracyOpProp, AccuracyOp, true);
}