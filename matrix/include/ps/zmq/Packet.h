//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_PACKET_H
#define MATRIX_PACKET_H

#include "Message.h"

namespace matrix {


    class Packet {
    public:


    private:
        Message header;
        Message content;
    };

}

#endif //MATRIX_PACKET_H
