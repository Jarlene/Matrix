//
// Created by Jarlene on 2017/7/21.
//

#ifndef MATRIX_BLOCKQUEUE_H
#define MATRIX_BLOCKQUEUE_H


#include <mutex>
#include <assert.h>
#include <list>

namespace matrix {
    template <class T>
    class BlockQueue {
    public:
        BlockQueue() {}

        void Put(const T &value) {
            {
                std::lock_guard<std::mutex> lock (mutex_);
                queue_.push_back (value);
            }
            condvar_.notify_all ();
        }

        T Take() {
            std::unique_lock<std::mutex> lock (mutex_);
            condvar_.wait (lock, [this]{return !queue_.empty ();});
            assert (!queue_.empty ());
            T front = queue_.front();
            queue_.pop_front ();
            return front;
        }

        long Size() {
            long size = 0;
            {
                std::unique_lock<std::mutex> lock (mutex_);
                size = queue_.size();
            }
            return size;
        }

        void Unique() {
            {
                std::unique_lock<std::mutex> lock (mutex_);
                queue_.sort();
                queue_.unique();
            }
        }

        void Clear() {
            {
                std::unique_lock<std::mutex> lock (mutex_);
                queue_.clear();
            }
        }

    private:
        BlockQueue (const BlockQueue& rhs);
        BlockQueue& operator = (const BlockQueue& rhs);

    private:
        std::mutex mutex_;
        std::condition_variable condvar_;
        std::list<T> queue_;
    };
}

#endif //MATRIX_BLOCKQUEUE_H
