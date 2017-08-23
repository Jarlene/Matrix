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
        int N = inputShapes[PREDICTION][0];
        int D = inputShapes[PREDICTION][1];
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
        Output<T>(OUT)[0] = T((correct * 1.0)/N);
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
    Operator* CreateOp<CPU>(AccuracyParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AccuracyOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(AccuracyParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AccuracyOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    AccuracyOpProp::AccuracyOpProp()  {
        param = new AccuracyParam(MatrixType::kFloat);

    }

    AccuracyOpProp::AccuracyOpProp(const MatrixType &type) {
        param = new AccuracyParam(type);
    }

    AccuracyOpProp::~AccuracyOpProp() {
        delete param;
    }

    void AccuracyOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        if (outShape.size() < 1) {
            Logger::Global()->Fatal("AccuracyOp  must has output shape. \n");
        }
        outShape.at(0).reShape(ShapeN(1));
    }

    Operator *AccuracyOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                             std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                             std::map<std::string, Any> &args) {
        param->args = args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void AccuracyOpProp::SwitchType( const MatrixType &type) {
        param->type = type;
    }
}