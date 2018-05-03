//
// Created by Jarlene on 2018/4/23.
//

#include "matrix/include/ps/Master.h"

namespace matrix {


    Master::Master() : BaseZMQ(){

    }

    Master::~Master()  {
        BaseZMQ::~BaseZMQ();
    }

    int Master::SendTo() {
        return 0;
    }

    int Master::ReceiveFrom() {
        return 0;
    }
}