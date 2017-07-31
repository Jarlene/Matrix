//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_BASEOPERATOR_H
#define MATRIX_BASEOPERATOR_H

#include "Operator.h"

namespace matrix {

    template <class Context>
    class BaseOperator : public Operator{
    public:
        BaseOperator() {}

        virtual bool Run() {
            Logger::Global()->Fatal("not Implementation");
            return false;
        }

        virtual void AsyncRun() {

        }

        virtual bool RunOnDevice() = 0;

        ~BaseOperator() noexcept  {

        }
        DISABLE_COPY_AND_ASSIGN(BaseOperator);
    };
}

#endif //MATRIX_BASEOPERATOR_H
