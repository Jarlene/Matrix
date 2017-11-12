//
// Created by Jarlene on 2017/7/31.
//

#include "matrix/include/op/AccuracyOp.h"


namespace matrix {


    template <class T, class Context>
    AccuracyOp<T, Context>::AccuracyOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool AccuracyOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    bool AccuracyOp<T, Context>::Run() {
        int N = inputShapes[PREDICTION]->At(0);
        int D = inputShapes[PREDICTION]->At(1);
        int correct = 0;
        for (int i = 0; i < N; ++i) {
            auto label = static_cast<int>(Input<T>(LABEL)[i]);
            auto label_pred = Input<T>(PREDICTION)[i * D + label];
            int index = 0;
            T max = Input<T>(PREDICTION)[i * D];
            for (int j = 1; j < D; ++j) {
                T pred = Input<T>(PREDICTION)[i * D + j];
                if (pred > max) {
                    index = j;
                }
            }
            if (label == index) {
                ++correct;
            }
        }
        assert(correct <= N);
        Output<T>()[0] = T((correct * 1.0)/N);
        return true;
    }

    template <class T, class Context>
    void AccuracyOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else if (Context::mode == RunMode::kGpu){
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    AccuracyOp<T, Context>::~AccuracyOp() {

    }




    template <>
    Operator* CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AccuracyOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AccuracyOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    AccuracyOpProp::AccuracyOpProp()  {
        param = new Parameter(MatrixType::kFloat);

    }

    AccuracyOpProp::AccuracyOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    AccuracyOpProp::~AccuracyOpProp() {
        delete param;
    }

    void AccuracyOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (outShape == nullptr) {
            Logger::Global()->Fatal("AccuracyOp  must has output shape. \n");
        }
        outShape->reShape(ShapeN(1));
    }

    Operator *AccuracyOpProp::CreateOperator(Context context, std::vector<Blob*> &input,Blob* output,
                                             std::vector<Shape*> &inShape, Shape *outShape,
                                             std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

}