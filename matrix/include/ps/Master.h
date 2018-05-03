//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_MASTER_H
#define MATRIX_MASTER_H


#include "zmq/BaseZMQ.h"

namespace matrix {


    /**
     * master monitor param servers and works, notify addresses to param servers and works
     */
    class Master : public BaseZMQ{
    public:
        Master();
        ~Master();

        int SendTo() override;

        int ReceiveFrom() override;
    };

}

#endif //MATRIX_MASTER_H
