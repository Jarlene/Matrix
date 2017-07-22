//
// Created by 郑珊 on 2017/7/21.
//

#ifndef MATRIX_BLOCKMAP_H
#define MATRIX_BLOCKMAP_H

#include <assert.h>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace matrix {

    template <typename Key, typename Value>
    class BlockMap {

    public:
        BlockMap() {

        }
        void Put(const Key &key, const Value & value) {
            {
                std::lock_guard<std::mutex> lock (mutex_);
                queue_[key]  = value;
            }
            condvar_.notify_all ();
        }

        Value &Get(const Key &key) {
            std::unique_lock<std::mutex> lock (mutex_);
            condvar_.wait (lock, [this]{return !queue_.empty ();});
            assert (!queue_.empty ());
            return queue_[key];
        }

        Value Take(const Key &key) {
            std::unique_lock<std::mutex> lock (mutex_);
            condvar_.wait (lock, [this]{return !queue_.empty ();});
            assert (!queue_.empty ());
            Value value = queue_[key];
            queue_.erase(key);
            return value;
        }

        long Size() {
            long size = 0;
            {
                std::lock_guard<std::mutex> lock (mutex_);
                size = queue_.size();
            }
            return size;
        }

        bool HasKey(const Key &key) {
            bool ok = false;
            {
                std::lock_guard<std::mutex> lock (mutex_);
                ok = (queue_.count(key) > 0);
            }
            return ok;
        }

        void Clear() {
            {
                std::lock_guard<std::mutex> lock (mutex_);
                queue_.clear();
            }
        }

        Value &operator[](const Key &key) {
            std::unique_lock<std::mutex> lock (mutex_);
            condvar_.wait (lock, [this]{return !queue_.empty ();});
            assert (!queue_.empty ());
            return queue_[key];
        }

    private:
        BlockMap (const BlockMap& rhs);
        BlockMap& operator = (const BlockMap& rhs);

    private:
        std::mutex mutex_;
        std::condition_variable condvar_;
        std::unordered_map<Key, Value> queue_;
    };

}
#endif //MATRIX_BLOCKMAP_H
