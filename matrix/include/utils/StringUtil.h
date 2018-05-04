//
// Created by Jarlene on 2018/4/16.
//

#ifndef MATRIX_STRINGUTIL_H
#define MATRIX_STRINGUTIL_H

#include <string>
#include <sstream>

using namespace std;


namespace matrix {

    template <class T>
    const string toString(T &val) {
        return std::to_string(val);
    }


    template <class T>
    const T fromString(string &val) {
        static_assert(std::is_arithmetic<T>::value, "T must is number");
        istringstream is(val);
        T t(0);
        is >> t;
        return t;
    }

}


#endif //MATRIX_STRINGUTIL_H
