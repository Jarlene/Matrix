//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LSTMOp.h"

namespace matrix {

    template <typename T>
    inline T sigmoid(T x) {
        return 1. / (1. + exp(-x));
    }

    template <typename T>
    inline T tanh(T x) {
        return 2. * sigmoid(2. * x) - 1.;
    }

    template <class T, class xpu>
    LSTMOp<T, xpu>::LSTMOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool LSTMOp<T, xpu>::Run() {
        int hide_num = GetArgValue<int>("hide_num");

        const int frame_size = hide_num * hide_num;

        const T *wi = Input<T>(WEIGHT);
        const T *wf = Input<T>(WEIGHT) + frame_size;
        const T *wc = Input<T>(WEIGHT) + 2 * frame_size;
        const T *wo = Input<T>(WEIGHT) + 3 * frame_size;

        T *c = InputNonConst<T>(C);
        T *h = InputNonConst<T>(H);

        const T * data = Input<T>(INPUT);
        int batch = inputShapes->at(INPUT)->At(0);
        int len = inputShapes->at(INPUT)->At(1);
        int dims = inputShapes->at(INPUT)->At(2);

        auto s = ShapeN(hide_num, hide_num);

        bool bias = GetArgValue<bool>("with_bias", true);

        int M = 1;
        int N = 1;
        int K = 1;




        T * out = Output<T>();

        if (bias) {
            const T* biasData = Input<T>(BIAS);
            for (int i = 0; i < batch; ++i) {
                // state->f = data * wf + biasData
                CPUGemm(NoTrans, NoTrans, M, N, K, T(1), data, wi, T(0), c);
                Add(N, c, biasData, c);
                Sigmoid(N, c, c);


                data += i * inputShapes->at(INPUT)->Size()/batch;
            }
        } else {



        }

        return true;
    }

    template <class T, class xpu>
    void LSTMOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool LSTMOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    LSTMOp<T, xpu>::~LSTMOp() {

    }



    template <class T, class xpu>
    bool LSTMOp<T, xpu>::VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        bool bias = GetArgValue<bool>("with_bias", true);
        int hide_num = GetArgValue<int>("hide_num");
        if (bias) {
            if (InputSize() < 4) {
                // for 8 params (wf, bf, wi, bi, wc, bc, wo, bo)
                Shape weight = ShapeN(hide_num, 4 * hide_num);
                Shape bias = ShapeN(7 * hide_num);
                func({&weight, &bias});
                return true;
            }
        } else {
            if (InputSize() < 3) {
                Shape weight = ShapeN(hide_num, 4 * hide_num);
                func({&weight});
                return true;
            }
        }
        return false;
    }

    template <class T, class xpu>
    bool LSTMOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        bool bias = GetArgValue<bool>("with_bias", true);
        int hide_num = GetArgValue<int>("hide_num");
        if ((bias && InputSize() < 4) || (!bias && InputSize() < 3)) {
            Shape weight = ShapeN(hide_num, 4 * hide_num);
            func({&weight, &weight});
            return true;
        }
        return false;
    }


    void LSTMOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        if(!param->args->count("hide_num")) {
            Logger::Global()->Fatal("LSTMOpProp InferShape==> need hide_num for output");
        }
        int hide_num = get<int>(param->args->at("hide_num"));
        int batch = inShape.at(0)->At(0);
        outShape->reShape(ShapeN(batch, hide_num));
    }

    INIT_OPERATOR_PROPERTY_CREATE(LSTMOpProp, LSTMOp, true);

}