//
// Created by 郑珊 on 2017/7/25.
//

#include "matrix/include/utils/OpRegistry.h"

namespace matrix {


    int OpRegistry::RegisterOp(const std::string &name, const OpPtr op) {
        if (opMap.count(name) == 0) {
            opMap[name] = op;
        }
        return 0;
    }

    const OpPtr OpRegistry::GetOp(const std::string &name) const {
        if (opMap.count(name)) {
            return opMap.at(name);
        }
        return nullptr;
    }
}