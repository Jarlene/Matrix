//
// Created by Jarlene on 2018/5/14.
//

#ifndef MATRIX_BARRIER_H
#define MATRIX_BARRIER_H


#include <mutex>
#include <thread>
#include <condition_variable>

namespace matrix {



    class Barrier {

    public:
        Barrier() {

        }


        explicit Barrier(int count) : flag_(count) {

        }


        inline void block() {
            std::unique_lock<std::mutex> glock(lock_);
            cond_.wait(glock, [this] {
                return flag_ == 0;
            });
        }


        template<class Func, class ...Args>
        inline void block_timeout(int milliseconds, Func &&func, Args &&... args) {
            std::thread timer_thread([this, milliseconds, func, args...] {
                std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
                func(args...);
                unblock();
            });
            timer_thread.detach();
        }

        inline void unblock() {
            std::unique_lock<std::mutex> glock(lock_);
            flag_--;
            assert(flag_ >= 0);
            cond_.notify_one();
        }

    private:
        int flag_{1};
        std::condition_variable cond_;
        std::mutex lock_;
    };
}




#endif //MATRIX_BARRIER_H
