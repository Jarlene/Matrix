//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_PACKET_H
#define MATRIX_PACKET_H

#include "Message.h"

namespace matrix {


    enum MsgType {
        RESPONSE = 0,
        REQUEST_HANDSHAKE,
        REQUEST_ACK,
        REQUEST_FIN,
        REQUEST_PUSH,
        REQUEST_PULL,
        HEARTBEAT,
        RESERVED
    };


    struct Packet {
        Packet() {

        }

        Packet(Packet &p) {
            this->header = std::move(p.header);
            this->content = std::move(p.content);
        }

        Packet &operator=(const Packet &) = delete;
        Packet(const Packet &) = delete;


        MsgType msg_type;
        size_t node_id;
        size_t to_node_id;
        size_t message_id;
        Message header;
        Message content;
    };

}

#endif //MATRIX_PACKET_H
