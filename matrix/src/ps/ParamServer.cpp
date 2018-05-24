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



    int ParamServer::registerMaster() {
//        Packet p;
//        p.msg_type = REQUEST_HANDSHAKE;
//        std::string local_addr = addr.toString();
//        p.header = Message(local_addr.c_str(), local_addr.length());
//        p.to_node_id = 0;
        return 0;
    }
}