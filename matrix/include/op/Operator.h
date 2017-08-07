//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_OPERATOR_H
#define MATRIX_OPERATOR_H

#include <map>
#include <vector>
#include "matrix/include/utils/MathTensor.h"
#include "matrix/include/api/MatrixType.h"
#include "matrix/include/utils/Logger.h"
#include "matrix/include/utils/Any.h"
#include "matrix/include/base/Blob.h"
#include "matrix/include/utils/OpRegistry.h"


#define BIND_DISPATCH(Method, ...) \
  return Method<cpu>(__VA_ARGS__); \

#define INPUT_TAG(first, ...)  \
  enum InputTags {first = 0, __VA_ARGS__} \

#define OUTPUT_TAG(first, ...)  \
  enum OutputTags {first = 0, __VA_ARGS__} \

#define SAME_FUNCTION(classname)  \
public:                           \
explicit classname##Op(classname##Param &param); \
virtual bool Run() override ; \
virtual void AsyncRun() override ; \
virtual ~classname##Op(); \
virtual bool RunOnDevice() override ; \


#define DISABLE_COPY_AND_ASSIGN(classname)                         \
private:                                                            \
  classname##Op(const classname##Op&) = delete;                              \
  classname##Op& operator=(const classname##Op&) = delete;


#define TYPE_SWITCH(type, DType, ...)         \
     switch(type) {                            \
          case  MatrixType::kInt:  \
            {                   \
               typedef int DType; \
               {__VA_ARGS__}      \
            }                     \
              break;               \
          case  MatrixType::kLong:  \
            {                   \
               typedef long DType; \
               {__VA_ARGS__}      \
            }                     \
              break;               \
          case  MatrixType::kFloat:  \
            {                   \
               typedef float DType; \
               {__VA_ARGS__}      \
            }                     \
              break;               \
          case  MatrixType::kDouble:  \
            {                   \
               typedef double DType; \
               {__VA_ARGS__}      \
            }                     \
              break;               \
          default:             \
               Logger::Global()->Fatal("switch type error %d", type); \
               break;       \
     }                          \

namespace matrix {

    class Operator {
    public:

        Operator() {

        }

        virtual ~Operator() {

        }

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


        virtual bool Run() {
            return false;
        }

        virtual void AsyncRun() {

        }

        virtual bool  InferShape() {
            return false;
        }

        virtual bool RunOnDevice() = 0;


    protected:
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

    struct Parameter {

        MatrixType  type;
        Parameter(MatrixType matrixType): type(matrixType) {

        }

    };



    class OperatorProperty {
    public:
        virtual void InferShape(std::vector<Shape> *inShape, std::vector<Shape> *outShape) const = 0;
        virtual Operator* CreateOperator(std::vector<Shape> *inShape, std::vector<Shape> *outShape) const {
            return nullptr;
        }
    };


}


#endif //MATRIX_OPERATOR_H
