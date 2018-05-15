//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_MESSAGE_H
#define MATRIX_MESSAGE_H

#include <zmq.h>

namespace matrix {

    class Message {
    public:
        Message();

        Message(char *buf, size_t size);

        Message(const Message &) = delete;

        Message &operator=(const Message &) = delete;

        Message &operator=(Message &&msg);

        ~Message();

        size_t size();

        char *data();

    private:
        zmq_msg_t msg;
    };

}

#endif //MATRIX_MESSAGE_H
