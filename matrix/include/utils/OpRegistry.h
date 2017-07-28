//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_OPREGISTER_H
#define MATRIX_OPREGISTER_H

#include <unordered_map>
#include "matrix/include/op/Operator.h"


#define STR(x) #x

#define CON_STR(name, type) STR(name##_##type)


#define INSTANTIATE_OPS(classname, name) \
  template class classname<float>; \
  template class classname<double>; \
  template class classname<int>; \
  template class classname<long>; \
  static int m_float_##name = matrix::OpRegistry::Global()->RegisterOp(CON_STR(name, float), std::make_shared<classname<float>>()); \
  static int m_double_##name = matrix::OpRegistry::Global()->RegisterOp(CON_STR(name, double), std::make_shared<classname<double>>()); \
  static int m_int_##name = matrix::OpRegistry::Global()->RegisterOp(CON_STR(name, int), std::make_shared<classname<int>>()); \
  static int m_long_##name = matrix::OpRegistry::Global()->RegisterOp(CON_STR(name, long), std::make_shared<classname<long>>()); \


namespace matrix {

    class Operator;

    using OpPtr = std::shared_ptr<Operator>;

    class OpRegistry {
    public:
        static OpRegistry *Global() {
            static OpRegistry registry;
            return &registry;
        }

        int RegisterOp(const std::string &name, const OpPtr op);

        const OpPtr GetOp(const std::string &name) const;

    private:
        std::unordered_map<std::string, OpPtr> opMap;
    };


}

#endif //MATRIX_OPREGISTER_H
