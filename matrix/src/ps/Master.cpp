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
}