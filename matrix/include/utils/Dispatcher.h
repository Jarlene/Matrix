//
// Created by Jarlene on 2018/4/14.
//

#ifndef MATRIX_DISPATCHER_H
#define MATRIX_DISPATCHER_H

#include <memory>

namespace matrix {


    template <class Construct,
              typename ...Args>
    std::shared_ptr<Construct> make(Args... args) {
        return std::make_shared<Construct>(args...);
    };

    template <class Func,
              class... Args>
    typename std::result_of<Func(Args...)>::type
    dispatch(Func func, Args... args) {
        return func(args...);
    };

    template<class Func,
             class OnSuccess,
             class OnError,
             class OnCompleted,
             typename... Args
    >
    void dispatch(Func func, OnSuccess success, OnError error, OnCompleted completed,  Args... args) {
        typedef typename std::result_of<Func(Args...)>::type ReturnType;
        try {
            ReturnType rs = func(args...);
            success(rs);
        } catch (std::exception &e) {
            error(e);
        }
        completed();
    };
}


#endif //MATRIX_DISPATCHER_H
