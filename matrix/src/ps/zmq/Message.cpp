//
// Created by Jarlene on 2018/4/23.
//


#include <string>
#include <cassert>
#include "matrix/include/ps/zmq/Message.h"

namespace matrix {
    Message::Message() {
        assert(zmq_msg_init(&msg) == 0);
    }

    Message::Message(char *buf, size_t size) {
        assert(zmq_msg_init_size(&msg, size) == 0);
        memcpy(zmq_msg_data(&msg), buf, size);
    }

    Message::~Message() {
        assert(0 == zmq_msg_close(&msg));
    }

    size_t Message::size() {
        return zmq_msg_size(&msg);
    }

    char *Message::data() {
        return (char *)zmq_msg_data(&msg);
    }

    Message &Message::operator=(Message &&msg) {
        if (this != &msg) {
            int res = zmq_msg_move(&this->msg, &msg.msg);
            assert(res == 0);
        }
        return *this;
    }
}