//
// Created by Jarlene on 2018/4/20.
//

#ifndef MATRIX_PARAMSERVER_H
#define MATRIX_PARAMSERVER_H


#include <matrix/include/ps/zmq/BaseZMQ.h>

namespace matrix {


    class ParamServer : public BaseZMQ {
    public:
        ParamServer();

        ~ParamServer() override;

        int registerMaster();
    };

}

#endif //MATRIX_PARAMSERVER_H
