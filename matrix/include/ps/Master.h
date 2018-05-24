//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_MASTER_H
#define MATRIX_MASTER_H

#include <vector>
#include "zmq/BaseZMQ.h"
#include "Worker.h"
#include "ParamServer.h"


namespace matrix {


    /**
     * master monitor param servers and works, notify addresses to param servers and works
     */
    class Master : public BaseZMQ {
    public:
        Master();

        Master(const std::string config);

        ~Master();

    private:

        /**
         * 启动master监听ps和worker上报机器地址和端口。
         * @return
         */
        int init();

    private:
        std::vector<Worker> workers;
        std::vector<ParamServer> pss;
    };

}

#endif //MATRIX_MASTER_H
