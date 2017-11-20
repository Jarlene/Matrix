//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_OPREGISTER_H
#define MATRIX_OPREGISTER_H

#include <unordered_map>
#include "matrix/include/op/Operator.h"


#define STR(x) #x

#define CON_STR(name, type) STR(name##_##type)


#define REGISTER_OP_PROPERTY(name, classname, ...) \
     const matrix::OpPtr op_##name = std::make_shared<matrix::classname>(__VA_ARGS__); \
     int name##_##classname = matrix::Registry::Global()->RegisterOp(STR(name), op_##name); \

#define GET_REGISTRY_OP_PROPERTY(name, type) \
     const matrix::OpPtr opPtr = matrix::Registry::Global()->GetOp(name, type); \

namespace matrix {

    class OperatorProperty;


    using OpPtr = std::shared_ptr<OperatorProperty>;


    class Registry {
    public:
        static Registry* Global() {
            static Registry registry;
            return &registry;
        }

        int RegisterOp(const std::string &name, const OpPtr op);

        const OpPtr GetOp(const std::string &name, const MatrixType &type = kFloat) const;

    private:
        std::unordered_map<std::string, OpPtr> opMap;
    };


}

#endif //MATRIX_OPREGISTER_H
