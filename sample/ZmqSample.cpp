//
// Created by Jarlene on 2018/3/15.
//
#ifdef USE_ZMQ
#include <zmq.h>
#endif
#include <cassert>
#include <unistd.h>
#include <iostream>
#include <thread>

using namespace std;

void client() {
#ifdef USE_ZMQ
    void *context = zmq_ctx_new();/// 创建一个新的环境
    assert(context != nullptr);

    int ret = zmq_ctx_set(context, ZMQ_MAX_SOCKETS, 1);/// 该环境中只允许有一个socket的存在
    assert(ret == 0);

    void *subscriber = zmq_socket(context, ZMQ_SUB);/// 创建一个订阅者
    assert(subscriber != nullptr);

    ret = zmq_connect(subscriber, "tcp://127.0.0.1:8100");/// 连接到服务器
    assert(ret == 0);

    ret = zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);/// 必须添加该语句对消息滤波，否则接受不到消息
    assert(ret == 0);

    char buf[16];/// 消息缓冲区

    for (int i = 0; i < 100; ++i) {
        ret = zmq_recv(subscriber, buf, 16, ZMQ_DONTWAIT);/// 接收消息，非堵塞式
        if (ret != -1) {
            /// 打印消息
            buf[ret] = '\0';
            printf("%s\n", buf);
        }
        sleep(1);
    }

    zmq_ctx_destroy(context);
#endif
}


void server() {
#ifdef USE_ZMQ
    void *ctx = zmq_ctx_new();
    assert(ctx != nullptr);


    int ret = zmq_ctx_set(ctx, ZMQ_MAX_SOCKETS, 10);/// 在该环境中最大只允许一个socket存在
    assert(ret == 0);

    void *publisher = zmq_socket(ctx, ZMQ_PUB);/// 创建一个发布者
    assert(publisher != nullptr);

    ret = zmq_bind(publisher, "tcp://127.0.0.1:8100");/// 绑定该发布到TCP通信
    assert(ret == 0);


    for (int i = 0; i < 100; ++i) {
        ret = zmq_send(publisher, "Hi,I'm server", 16, 0);/// 发送消息

        printf("%d\n", ret);
        sleep(1);
    }

    zmq_ctx_destroy(ctx);
#endif
}


int main(int argc, char *argv[]) {

    std::thread t1(server);
    std::thread t2(client);

    t1.join();
    t2.join();
    return 0;
}