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

    const OpPtr Registry::GetOp(const std::string &name) const {
        if (opMap.count(name)) {
            return opMap.at(name);
        }
        return nullptr;
    }
}