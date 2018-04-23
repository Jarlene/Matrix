//
// Created by Jarlene on 2018/4/20.
//

#ifndef MATRIX_RING_H
#define MATRIX_RING_H


#include <vector>
#include <map>
#include <sstream>
#include "matrix/include/utils/Hash.h"

namespace matrix {

    template<class T>
    class Ring {
    public:
        Ring(const std::vector<T> names) {
            for (auto &name : names) {
                AddServer(name);
            }
        }

        Ring(const std::vector<T> names, int cp) : replicas(cp) {
            for (auto &name : names) {
                AddServer(name);
            }
        }


        void AddServer(const T &name) {
            std::ostringstream oss;
            oss << name;
            for (int i = 0; i < replicas; ++i) {
                size_t hash = Hash()(oss.str() + ":" + std::to_string(i));
                dict[hash] = name;
                list.push_back(hash);
            }
            std::sort(list.begin(), list.end());
        }


        void RemoveServer(const T &name) {
            std::ostringstream oss;
            oss << name;
            for (int i = 0; i < replicas; ++i) {
                size_t hash = Hash()(oss.str() + ":" + std::to_string(i));
                dict.erase(hash);
                auto it = std::find(list.begin(), list.end(), hash);
                if (it != list.end()) {
                    list.erase(it);
                }
            }
        }


        template<class P>
        T GetServer(const P &name) {
            std::ostringstream oss;
            oss << name;
            size_t hash = Hash()(oss.str());
            auto severHash = ring_search(list, hash);
            return dict[severHash];
        }

    private:
        int replicas = 32;
        std::vector<size_t> list;
        std::map<size_t, T> dict;

    };
}
#endif //MATRIX_RING_H
