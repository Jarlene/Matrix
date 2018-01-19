//
// Created by Jarlene on 2017/11/27.
//

#include "matrix/include/op/BatchNormalizationOp.h"


namespace matrix {

    template <class T, class xpu>
    BatchNormalizationOp<T, xpu>::BatchNormalizationOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool BatchNormalizationOp<T, xpu>::Run() {

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
            Shape shape;
            func({&shape, &shape});
            return true;
        }
        return false;
    }

    void BatchNormalizationOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        if(!param->args->count("hide_num")) {
            Logger::Global()->Fatal("LSTMOpProp InferShape==> need hide_num for out put");
        }
        int hide_num = get<int>(param->args->at("hide_num"));
        outShape->reShape(ShapeN(inShape.at(0)->At(0), hide_num));
    }

    INIT_OPERATOR_PROPERTY_CREATE(BatchNormalizationOpProp, BatchNormalizationOp, true);

}