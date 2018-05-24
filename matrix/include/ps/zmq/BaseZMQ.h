//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_BASEZMQ_H
#define MATRIX_BASEZMQ_H


#ifndef USE_ZMQ
#pragma message("Error: Parameters servers need zeroMQ please set option use_zmp on")
#endif

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <cassert>
#include <zmq.h>
#include <string>
#include <arpa/inet.h>
#include "Packet.h"
#include "matrix/include/utils/StringUtil.h"

namespace matrix {


    struct Addr {
        std::string addr;
        size_t port;

        const std::string toString() const {
            return "tcp://" + addr + std::to_string(port);
        }

        Addr() {

        };

        Addr(Message &msg) {
            std::string ip = std::string(msg.data());

        }

    };

    class BaseZMQ {
    public:
        BaseZMQ() {
            zmq_ctx = zmq_ctx_new();
            assert(zmq_ctx != nullptr);
            zmq_skt = zmq_socket(zmq_ctx, ZMQ_PULL);
            assert(zmq_skt != nullptr);
            this->addr = new Addr();
            addr->addr = getLocalIp();
            addr->port = getRandomPort();
        }

        virtual ~BaseZMQ() {
            zmq_ctx_destroy(zmq_ctx);
            if (zmq_skt) {
                zmq_close(zmq_skt);
                zmq_skt = nullptr;
            }
            if (addr) {
                delete addr;
            }
        }


        virtual int SendTo(const Addr &addr, Packet *p) {
            int retry_count = 5;
            int ret = -1;
            while (retry_count--) {
                ret = zmq_bind(zmq_skt, addr.toString().c_str());
                if (ret == 0) {
                    break;
                }
            }
            if (ret != 0) {
                return -1;
            }
            ret = -1;
            retry_count = 5;
            while (retry_count--) {
                ret = zmq_send(zmq_skt, p->header.data(), p->header.size(), ZMQ_SNDMORE);
                if (ret != p->header.size()) {
                    continue;
                }
                ret = zmq_send(zmq_skt, p->content.data(), p->content.size(), 0);
                if (ret == 0) {
                    break;
                }
            }
            if (ret != 0) {
                return -1;
            }
            return 0;
        };

        virtual int ReceiveFrom(const Addr &addr, Packet *p) {
            int retry_count = 5;
            int ret = -1;
            while (retry_count--) {
                ret = zmq_bind(zmq_skt, addr.toString().c_str());
                if (ret == 0) {
                    break;
                }
            }
            if (ret != 0) {
                return -1;
            }
            ret = -1;
            retry_count = 5;
            while (retry_count--) {
                ret = zmq_recv(zmq_skt, p->header.data(), p->header.size(), ZMQ_RCVMORE);
                if (ret != p->header.size()) {
                    continue;
                }
                ret = zmq_recv(zmq_skt, p->content.data(), p->content.size(), 0);
                if (ret == 0) {
                    break;
                }
            }
            if (ret != 0) {
                return -1;
            }
            return 0;
        }


    protected:
        int registerRouter(size_t node_id, Addr &&addr) {
            return 0;
        }

        std::string getLocalIp() {
            struct ifaddrs *ifAddrStruct = nullptr;
            struct ifaddrs *ifa = nullptr;
            void *tmpAddrPtr = nullptr;
            std::string local_ip;

            getifaddrs(&ifAddrStruct);
            for (ifa = ifAddrStruct; ifa; ifa = ifa->ifa_next) {
                if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) { // IPv4 Address
                    tmpAddrPtr = &((struct sockaddr_in *) ifa->ifa_addr)->sin_addr;
                    char address[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, tmpAddrPtr, address, INET_ADDRSTRLEN);
                    if (strcmp(address, "127.0.0.1") != 0 &&
                        strcmp(address, "0.0.0.0") != 0) {
                        local_ip = address;
                    }
                }
            }
            if (ifAddrStruct)
                freeifaddrs(ifAddrStruct);
            return local_ip;
        }

        size_t getRandomPort() {
            return static_cast<size_t>(1024 + rand() % (65536 - 1024));
        }

    protected:
        void *zmq_ctx;
        void *zmq_skt;
        Addr *addr;

    };
}


#endif //MATRIX_BASEZMQ_H
