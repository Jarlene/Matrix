//
// Created by Jarlene on 2017/8/25.
//

#ifndef MATRIX_REDUCEOP_H
#define MATRIX_REDUCEOP_H

#include "matrix/include/base/Tensor.h"
#include "matrix/include/utils/Math.h"

namespace matrix {

    template <class T>
    void Sum(const Tensor<T>& tensor, int dim, Tensor<T>& reduce) {
        const T* tensorData = tensor.Data();
        T* reduceData = reduce.MutableData();
        assert(tensorData);
        assert(reduceData);
        if (tensor.Rank() == 1) {
            T temp = T(0);
            auto func = [&tensorData, &temp](int i) {
                temp += tensorData[i];
            };
            Reduce<T>(tensor.Size(), func);
            reduceData[0] = temp;
        } else if (tensor.Rank() > 1) {
            auto s = tensor.GetShape();
            int strideOut = s.StrideExclude(dim);
            int strideIn = s.StrideInclude(dim);
            int shapeDim = s.At(dim);
            auto func = [&tensorData, &reduceData, &strideOut, &strideIn, &shapeDim](int i) {
                T temp = T(0);
                int fi = (i/strideOut)*strideIn + i % strideOut;
                int fj = 0;
                for (int j = 0; j < shapeDim; ++j) {
                    temp += tensorData[fi + fj];
                    fj += strideOut;
                }
                reduceData[i] = temp;
            };
            Reduce<T>(reduce.Size(), func);
        }
    }

    template <class T>
    void SumAdd(const Tensor<T>& tensor, int dim, Tensor<T>& reduce) {
        const T* tensorData = tensor.Data();
        T* reduceData = reduce.MutableData();
        assert(tensorData);
        assert(reduceData);
        if (tensor.Rank() == 1) {
            T temp = T(0);
            auto func = [&tensorData, &temp](int i) {
                temp += tensorData[i];
            };
            Reduce<T>(tensor.Size(), func);
            reduceData[0] += temp;
        } else if (tensor.Rank() > 1) {
            auto s = tensor.GetShape();
            int strideOut = s.StrideExclude(dim);
            int strideIn = s.StrideInclude(dim);
            int shapeDim = s.At(dim);
            auto func = [&tensorData, &reduceData, &strideOut, &strideIn, &shapeDim](int i) {
                T temp = T(0);
                int fi = (i/strideOut)*strideIn + i % strideOut;
                int fj = 0;
                for (int j = 0; j < shapeDim; ++j) {
                    temp += tensorData[fi + fj];
                    fj += strideOut;
                }
                reduceData[i] += temp;
            };
            Reduce<T>(reduce.Size(), func);
        }
    }


    template <typename T>
    void Mean(const Tensor<T>& tensor, int dim, Tensor<T>& reduce) {
        const T* tensorData = tensor.Data();
        T* reduceData = reduce.MutableData();
        assert(tensorData);
        assert(reduceData);
        if (tensor.Rank() == 1) {
            T temp = T(0);
            auto func = [&tensorData, &temp](int i) {
                temp += tensorData[i];
            };
            Reduce<T>(tensor.Size(), func);
            reduceData[0] = temp/tensor.GetShape().At(0);
        } else if (tensor.Rank() > 1) {
            auto s = tensor.GetShape();
            int strideOut = s.StrideExclude(dim);
            int strideIn = s.StrideInclude(dim);
            int shapeDim = s.At(dim);

            auto func = [&tensorData, &reduceData, &strideOut, &strideIn, &shapeDim](int i) {
                T temp = T(0);
                int fi = (i/strideOut)*strideIn + i % strideOut;
                int fj = 0;
                for (int j = 0; j < shapeDim; ++j) {
                    temp += tensorData[fi + fj];
                    fj += strideOut;
                }
                reduceData[i] = temp/shapeDim;
            };
            Reduce<T>(reduce.Size(), func);
        }
    }


    template <typename T>
    void ASum(const Tensor<T>& tensor, int dim, Tensor<T>& reduce) {
        const T* tensorData = tensor.Data();
        T* reduceData = reduce.MutableData();
        assert(tensorData);
        assert(reduceData);
        if (tensor.Rank() == 1) {
            T temp = T(0);
            auto func = [&tensorData, &temp](int i) {
                temp += abs(tensorData[i]);
            };
            Reduce<T>(tensor.Size(), func);
            reduceData[0] = temp;
        } else if (tensor.Rank() > 1) {
            auto s = tensor.GetShape();
            int strideOut = s.StrideExclude(dim);
            int strideIn = s.StrideInclude(dim);
            int shapeDim = s.At(dim);
            auto func = [&tensorData, &reduceData, &strideOut, &strideIn, &shapeDim](int i) {
                T temp = T(0);
                int fi = (i/strideOut)*strideIn + i % strideOut;
                int fj = 0;
                for (int j = 0; j < shapeDim; ++j) {
                    temp += abs(tensorData[fi + fj]);
                    fj += strideOut;
                }
                reduceData[i] = temp;
            };
            Reduce<T>(reduce.Size(), func);
        }
    }

    template <typename T>
    void Normal2(const Tensor<T>& tensor, int dim, Tensor<T>& reduce) {
        const T* tensorData = tensor.Data();
        T* reduceData = reduce.MutableData();
        assert(tensorData);
        assert(reduceData);
        if (tensor.Rank() == 1) {
            T temp = T(0);
            auto func = [&tensorData, &temp](int i) {
                temp += tensorData[i] * tensorData[i];
            };
            Reduce<T>(tensor.Size(), func);
            reduceData[0] = sqrt(temp);
        } else if (tensor.Rank() > 1) {
            auto s = tensor.GetShape();
            int strideOut = s.StrideExclude(dim);
            int strideIn = s.StrideInclude(dim);
            int shapeDim = s.At(dim);
            auto func = [&tensorData, &reduceData, &strideOut, &strideIn, &shapeDim](int i) {
                T temp = T(0);
                int fi = (i/strideOut)*strideIn + i % strideOut;
                int fj = 0;
                for (int j = 0; j < shapeDim; ++j) {
                    temp += tensorData[fi + fj] * tensorData[fi + fj];
                    fj += strideOut;
                }
                reduceData[i] = sqrt(temp);
            };
            Reduce<T>(reduce.Size(), func);
        }
    }

    template <typename T>
    void Max(const Tensor<T>& tensor, int dim, Tensor<T>& reduce, Tensor<int> *indexTensor = nullptr) {
        const T* tensorData = tensor.Data();
        T* reduceData = reduce.MutableData();
        assert(tensorData);
        assert(reduceData);
        if (tensor.Rank() == 1) {
            T temp = tensorData[0];
            int idx = 0;
            auto func = [&tensorData, &temp, &idx, &indexTensor](int i) {
                if (temp < tensorData[i]) {
                    temp = tensorData[i];
                    idx = i;
                }
            };
            Reduce<T>(tensor.Size(), func);
            reduceData[0] = temp;
            if (indexTensor != nullptr) {
                indexTensor->MutableData()[0] = idx;
            }
        } else if (tensor.Rank() > 1) {
            auto s = tensor.GetShape();
            int strideOut = s.StrideExclude(dim);
            int strideIn = s.StrideInclude(dim);
            int shapeDim = s.At(dim);
            auto func = [&tensorData, &reduceData, &strideOut, &strideIn, &shapeDim, &indexTensor](int i) {
                int fi = (i/strideOut)*strideIn + i % strideOut;
                int fj = 0;
                T temp = tensorData[fi];
                int idx = fi;
                for (int j = 0; j < shapeDim; ++j) {
                    if (temp < tensorData[fi + fj]) {
                        temp = tensorData[fi +fj];
                        idx = fi +fj;
                    }
                    fj += strideOut;
                }
                reduceData[i] = temp;
                if (indexTensor != nullptr) {
                    indexTensor->MutableData()[i] = idx;
                }
            };
            Reduce<T>(reduce.Size(), func);
        }
    }

    template <typename T>
    void Min(const Tensor<T>& tensor, int dim, Tensor<T>& reduce,  Tensor<int> *indexTensor = nullptr) {
        const T* tensorData = tensor.Data();
        T* reduceData = reduce.MutableData();
        assert(tensorData);
        assert(reduceData);
        if (tensor.Rank() == 1) {
            T temp = tensorData[0];
            int idx = 0;
            auto func = [&tensorData, &temp, &idx, &indexTensor](int i) {
                if (temp > tensorData[i]) {
                    temp = tensorData[i];
                    idx = i;
                }
            };
            Reduce<T>(tensor.Size(), func);
            reduceData[0] = temp;
            if (indexTensor != nullptr) {
                indexTensor->MutableData()[0] = idx;
            }
        } else if (tensor.Rank() > 1) {
            auto s = tensor.GetShape();
            int strideOut = s.StrideExclude(dim);
            int strideIn = s.StrideInclude(dim);
            int shapeDim = s.At(dim);
            auto func = [&tensorData, &reduceData, &strideOut, &strideIn, &shapeDim, &indexTensor](int i) {
                int fi = (i/strideOut)*strideIn + i % strideOut;
                int fj = 0;
                T temp = tensorData[fi];
                int idx = fi;
                for (int j = 0; j < shapeDim; ++j) {
                    if (temp > tensorData[fi + fj]) {
                        temp = tensorData[fi +fj];
                        idx = fi +fj;
                    }
                    fj += strideOut;
                }
                reduceData[i] = temp;
                if (indexTensor != nullptr) {
                    indexTensor->MutableData()[i] = idx;
                }
            };
            Reduce<T>(reduce.Size(), func);
        }
    }

    template <typename T>
    void Broadcast(const Tensor<T>& tensor, int dim, Tensor<T>& reduce) {

    }

}

#endif //MATRIX_REDUCEOP_H
