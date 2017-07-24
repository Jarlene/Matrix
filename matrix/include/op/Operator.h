//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_OPERATOR_H
#define MATRIX_OPERATOR_H

#include <map>
#include "matrix/include/utils/Any.h"
#include "matrix/include/base/Blob.h"
namespace matrix {

    class Operator {
    public:
        inline bool HasArg(const std::string &name) {
            return args.count(name) > 0;
        }

        template <class T>
        inline T getArgValue(const std::string & name, const T &default_value) {
            if (args.count(name)) {
                return get(args.at(name));
            }
            return default_value;
        }


        template <class T>
        inline const T& Input(int idx) {
            return input.at(idx).Get<T>();
        }

        template <class T>
        inline T* Output(int idx) {
            return output.at(idx)->GetMutable();
        }


        virtual bool Run();


        virtual void AsyncRun();


    private:
        std::map<std::string, Any> args;
        std::vector<Blob> input;
        std::vector<Blob*> output;
    };

}


#endif //MATRIX_OPERATOR_H
