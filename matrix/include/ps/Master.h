//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_MASTER_H
#define MATRIX_MASTER_H


#include "zmq/BaseZMQ.h"

namespace matrix {


    class Master : public BaseZMQ{
    public:
        Master();
        ~Master();
    };

}

#endif //MATRIX_MASTER_H
