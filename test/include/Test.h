//
// Created by Jarlene on 2017/7/24.
//

#ifndef MATRIX_TEST_H
#define MATRIX_TEST_H
namespace matrix {
    int main(int argc, char **argv) {
        testing::InitGoogleTest(&argc, argv);
        return 0;
    }

    template <typename Dtype>
    inline int checkArrayEqual(const Dtype* arr1, const Dtype* arr2, const int dim) {
        for(int i=0; i<dim; ++i) {
            EXPECT_EQ(arr1[i], arr2[i]);
        }
        return 0;
    }

    template <>
    inline int checkArrayEqual<float>(const float* arr1, const float* arr2, const int dim) {
        for(int i=0; i<dim; ++i) {
            EXPECT_FLOAT_EQ(arr1[i], arr2[i]);
        }
        return 0;
    }

    template <>
    inline int checkArrayEqual<double>(const double* arr1,const  double* arr2, const int dim) {
        for(int i=0; i<dim; ++i) {
            EXPECT_DOUBLE_EQ(arr1[i], arr2[i]);
        }
        return 0;
    }

    template <>
    inline int checkArrayEqual<int>(const int * arr1, const int * arr2, const int dim) {
        for (int i = 0; i < dim; ++i) {
            EXPECT_EQ(arr1[i], arr2[i]);
        }
        return 0;
    }

    template <>
    inline int checkArrayEqual<long>(const long * arr1, const long * arr2, const int dim) {
        for (int i = 0; i < dim; ++i) {
            EXPECT_EQ(arr1[i], arr2[i]);
        }
        return 0;
    }
}
#endif //MATRIX_TEST_H
