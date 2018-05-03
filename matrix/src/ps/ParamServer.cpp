//
// Created by Jarlene on 2018/5/3.
//

#include "matrix/include/ps/ParamServer.h"

namespace matrix {


    ParamServer::ParamServer() : BaseZMQ() {

    }

    ParamServer::~ParamServer() {
        BaseZMQ::~BaseZMQ();
    }

    int ParamServer::SendTo() {
        return 0;
    }

    int ParamServer::ReceiveFrom() {
        return 0;
    }
}