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

    /**
     * i_t & =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+b_i)\\
     * f_t & = \sigma(W_{fx}x_t+W_{fh}h_{t-1}+b_f+1)\\
     * o_t & = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+b_o)\\
     * \tilde{c_t} & = \tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)\\
     * c_t & = c_{t-1}\circ f_t + \tilde{c_t}\circ i_t\\
     * h_t & = \tanh(c_t)\circ o_t\\
     * @tparam T
     * @tparam xpu
     * @return
     */
    template <class T, class xpu>
    bool LSTMOp<T, xpu>::Run() {
        int hide_num = GetArgValue<int>("hide_num");

        T *c = InputNonConst<T>(C);
        T *h = InputNonConst<T>(H);

        const T * data = Input<T>(INPUT);
        int batch = inputShapes->at(INPUT)->At(0);
        int len = inputShapes->at(INPUT)->At(1);
        int dims = inputShapes->at(INPUT)->At(2);

        auto s = ShapeN(hide_num, hide_num);
        auto sqs = ShapeN(len, dims);

        bool bias = GetArgValue<bool>("with_bias", false);


        const T *wix = Input<T>(WIX);
        const T *wih = Input<T>(WIH);
        const T *wfx = Input<T>(WFX);
        const T *wfh = Input<T>(WFH);
        const T *wox = Input<T>(WOX);
        const T *woh = Input<T>(WOH);
        const T *wcx = Input<T>(WCX);
        const T *wch = Input<T>(WCH);
        T *out = Output<T>();
        if (bias) {
            const T *bi = Input<T>(BI);
            const T *bf = Input<T>(BF);
            const T *bo = Input<T>(BO);
            const T *bc = Input<T>(BC);
            for (int i = 0; i < batch; ++i) {
                Tensor<T> seq(data + i * sqs.Size(), sqs);
                Tensor<T> wx(wix, s);
                Tensor<T> it(out, *outputShape);
                if (i == 0) {
                    MatrixMul(seq, false, wx, false, it);
                } else {

                }
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
        bool bias = GetArgValue<bool>("with_bias", false);
        int hide_num = GetArgValue<int>("hide_num");
        int size = InputShape(0)->At(1);
        Shape weight = ShapeN(size, hide_num);
        if (bias) {
            if (InputSize() < 3) {
                // for 8 params (wf, bf, wi, bi, wc, bc, wo, bo)
                Shape bias = ShapeN(hide_num);
                func({&weight, &weight, &weight, &weight, &weight, &weight, &weight, &weight, &bias, &bias, &bias , &bias});
                return true;
            }
        } else {
            if (InputSize() < 2) {
                func({&weight, &weight, &weight, &weight, &weight, &weight, &weight, &weight});
                return true;
            }
        }
        return false;
    }

    template <class T, class xpu>
    bool LSTMOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        bool bias = GetArgValue<bool>("with_bias", false);
        int hide_num = GetArgValue<int>("hide_num");
        if ((bias && InputSize() < 5) || (!bias && InputSize() < 4)) {
            Shape weight = ShapeN(hide_num, 4 * hide_num);
            func({&weight, &weight, &weight, &weight});
            return true;
        }
        return false;
    }


    void LSTMOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        bool bias = param->GetArgValue<bool>("with_bias", false);
        if ((inShape.size() < 4 && bias) || (inShape.size() < 3 && !bias)) {
            return;
        }
        int hide_num = param->GetArgValue<int>("hide_num");
        int batch = inShape.at(0)->At(0);
        outShape->reShape(ShapeN(batch, hide_num));
    }

    INIT_OPERATOR_PROPERTY_CREATE(LSTMOpProp, LSTMOp, true);

}