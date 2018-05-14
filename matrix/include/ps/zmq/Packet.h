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


    class Packet {
    public:
        void* data() {
            // todo::has a good method for this?
            void * result = malloc(len());
            memcpy(result, header.data(), header.size());
            memcpy(result + header.size(), content.data(), content.size());
            return result;
        }
        size_t len() {
            return header.size() + content.size();
        }


        void *headData() {
            header.data();
        }

        size_t headLen() {
            header.size();
        }

        void *contentData() {
            content.data();
        }

        size_t contentLen() {
            content.size();
        }

    private:
        MsgType msg_type;

        size_t node_id;
        size_t to_node_id;
        size_t message_id;

        Message header;
        Message content;
    };

}

#endif //MATRIX_PACKET_H
