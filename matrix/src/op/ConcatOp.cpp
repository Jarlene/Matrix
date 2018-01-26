//
// Created by Jarlene on 2017/11/20.
//

#include "matrix/include/op/ConcatOp.h"


namespace matrix {

    template <class T, class xpu>
    ConcatOp<T, xpu>::ConcatOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool ConcatOp<T, xpu>::Run() {
        T *out = Output<T>();
        int total = 0;
        for (int i = 0; i < InputSize(); ++i) {
            int size = inputShapes->at(i)->Size();
            CPUCopy(size, Input<T>(i), 1, out + total, 1);
            total += size;
        }
        return true;
    }

    template <class T, class xpu>
    void ConcatOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool ConcatOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    ConcatOp<T, xpu>::~ConcatOp() {

    }



    void ConcatOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        int rank = inShape.at(0)->Rank();
        for (int i = 1; i < inShape.size(); ++i) {
            assert(inShape[i]->Rank() == rank);
        }
        int axis = param->GetArgValue<int>("axis", -1);
        if (axis == -1) {
            int size = inShape.size();
            for (int j = 0; j < rank; ++j) {
                for (int i = 0; i < size - 1; ++i) {
                    for (int k = i + 1; k < size; ++k) {
                        if (inShape[i]->At(j) != inShape[k]->At(j)) {
                            if (axis != -1 && j != axis) {
                                Logger::Global()->Fatal("ConcatOpProp InferShape concat axis not match!");
                            }
                            axis = j;
                        }
                    }
                }
            }
        }
        if (axis == -1) {
            Logger::Global()->Fatal("ConcatOpProp InferShape can not find which axis to concat!");
        }
        for (int i = 0; i < rank; ++i) {
            if (i == axis) {
                int total = 0;
                for (int j = 0; j < inShape.size(); ++j) {
                    total += inShape[j]->At(i);
                }
                outShape->Append(total);
            } else {
                outShape->Append(inShape[0]->At(i));
            }
        }
    }

    INIT_OPERATOR_PROPERTY_CREATE(ConcatOpProp, ConcatOp, true);


}