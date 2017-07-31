//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_OPTIONAL_H
#define MATRIX_OPTIONAL_H

#include <typeinfo>
#include <type_traits>

namespace matrix {


    template <class T>
    class Optional {
    public:
        Optional() : isNone(true) {

        }

        explicit Optional(const T& value) {
            isNone = false;
            new (&val) T(value);
        }

        ~Optional() {
            if (!isNone) {
                reinterpret_cast<T*>(&val)->~T();
            }
        }

        void swap(Optional<T>& other) {
            std::swap(val, other.val);
            std::swap(isNone, other.isNone);
        }

        Optional<T>& operator=(const T& value) {
            (Optional<T>(value)).swap(*this);
            return *this;
        }

        Optional<T>& operator=(const Optional<T> &other) {
            (Optional<T>(other)).swap(*this);
            return *this;
        }

        T& operator*() {
            return *reinterpret_cast<T*>(&val);
        }

        const T& operator*() const {
            return *reinterpret_cast<const T*>(&val);
        }

        const T& value() const {
            if (isNone) {
                throw std::logic_error("bad optional access");
            }
            return *reinterpret_cast<const T*>(&val);
        }

        explicit operator bool() const {
            return !isNone;
        }

    private:
        bool isNone;
        typename std::aligned_storage<sizeof(T), alignof(T)>::type val;
    };

}

#endif //MATRIX_OPTIONAL_H
