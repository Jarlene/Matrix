//
// Created by Jarlene on 2018/4/23.
//

#include <cassert>
#include "matrix/include/ps/Master.h"

namespace matrix {


    Master::Master() : BaseZMQ(){
        init();
    }

    Master::Master(const std::string config) : BaseZMQ(){

        init();
    }

    Master::~Master()  {
        BaseZMQ::~BaseZMQ();
    }



    int Master::init() {


        return 0;
    }
}