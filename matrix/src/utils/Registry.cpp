//
// Created by Jarlene on 2017/7/25.
//

#include "matrix/include/utils/Registry.h"

namespace matrix {


    int Registry::RegisterOp(const std::string &name, const OpPtr op) {
        if (opMap.count(name) == 0) {
            opMap[name] = op;
        }
        return 0;
    }

    const OpPtr Registry::GetOp(const std::string &name, const MatrixType &type) const {
        if (opMap.count(name)) {
            auto op_prop = opMap.at(name);
            op_prop->SwitchType(type);
            return op_prop;
        }
        return nullptr;
    }
}