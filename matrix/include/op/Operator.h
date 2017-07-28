//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_OPERATOR_H
#define MATRIX_OPERATOR_H

#include <map>
#include <vector>
#include "matrix/include/utils/Any.h"
#include "matrix/include/base/Blob.h"
#include "matrix/include/utils/OpRegistry.h"


#define DISABLE_COPY_AND_ASSIGN(classname)                         \
private:                                                            \
  classname(const classname&) = delete;                              \
  classname& operator=(const classname&) = delete;

namespace matrix {

    class Operator {
    public:
        inline bool HasArg(const std::string &name) {
            return args.count(name) > 0;
        }

        template <class T>
        inline T getArgValue(const std::string & name, const T &default_value) {
            if (args.count(name)) {
                return get<T>(args.at(name));
            }
            return default_value;
        }


        template <class T>
        inline const T& Input(int idx) {
            return input.at(idx).Get<T>();
        }

        template <class T>
        inline T* Output(int idx) {
            return output.at(idx)->GetMutable<T>();
        }

        inline const std::vector<Blob> Inputs() const {
            return input;
        }

        inline const std::vector<Blob*> Outputs() const {
            return output;
        }


        virtual bool Run();


        virtual void AsyncRun();


    private:
        std::map<std::string, Any> args;
        std::vector<Blob> input;
        std::vector<Blob*> output;
    };


    class State {
    public:
        ~State() {
            clear();
        }

        virtual void clear();
    };
}


#endif //MATRIX_OPERATOR_H
