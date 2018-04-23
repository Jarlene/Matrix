//
// Created by Jarlene on 2018/4/20.
//

#ifndef MATRIX_HASH_H
#define MATRIX_HASH_H

#include <string>
#include <vector>

namespace matrix {

    template <class T>
    struct Hash {
        size_t  operator()(const T &t) const {
            return std::hash<T>()(t);
        }
    };


    template<class T>
    size_t ring_search(std::vector<T> data, T key) {
        if (key < data[0] || key > data[data.size() - 1]) return 0;
        size_t s = 0, e = data.size() - 1;
        size_t m;
        while (s <= e) {
            m = s + (e - s) / 2;
            if (data[m] < key) {
                s = m + 1;
            } else if (data[m] > key) {
                e = m - 1;
            } else {
                return m;
            }
        }
        return data[m] < key ? m + 1 : m;
    }
}

#endif //MATRIX_HASH_H
