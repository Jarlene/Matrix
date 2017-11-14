//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_OPERATOR_H
#define MATRIX_OPERATOR_H

#include <map>
#include <vector>
#include <initializer_list>
#include "matrix/include/base/Blob.h"
#include "matrix/include/utils/MathTensor.h"
#include "matrix/include/api/MatrixType.h"
#include "matrix/include/utils/Logger.h"
#include "matrix/include/utils/Any.h"
#include "matrix/include/utils/Registry.h"
#include "matrix/include/utils/ProduceShape.h"






#define INIT_OPERATOR_PROPERTY(classname) \
public: \
   classname(); \
   classname(const MatrixType &type); \
   ~classname();  \
   virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape); \
   virtual Operator* CreateOperator(Context context,  \
                                 std::vector<Shape*> *inShape, Shape *outShape, \
                                 std::map<std::string, Any> &args) ; \

#define INIT_PARAMS  \
    this->inputShapes = param.inputShapes;\
    this->outputShape = param.outShape; \
    this->args = param.args; \



#define INPUT_TAG(first, ...)  \
  enum InputTags {first = 0, __VA_ARGS__} \


#define SAME_FUNCTION(classname)  \
public:                           \
explicit classname##Op(matrix::Parameter &param); \
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
               Logger::Global()->Fatal("switch type error %d\n", type); \
               break;       \
     }                          \



#define CREATE_OPERATOR(context, param, name, ...) \
   Operator *op = nullptr;  \
   TYPE_SWITCH(param->type, DType, {  \
      if (context.mode == kCpu) { \
         op = new name<DType, CPU>(*param); \
      } else if (context.mode == kGpu) { \
         op = new name<DType, GPU>(*param); \
      }\
      {__VA_ARGS__} \
   }) \
   return op; \

#define INIT_OPERATOR_PROPERTY_CREATE(classname, name, memory) \
    classname::classname()  {  \
        param = new Parameter(MatrixType::kFloat);\
    }\
    classname::classname(const MatrixType &type) { \
        param = new Parameter(type); \
    } \
    classname::~classname() { \
        delete param; \
    }  \
    Operator *classname::CreateOperator(Context context, \
                                    std::vector<Shape *> *inShape, Shape *outShape, \
                                    std::map<std::string, Any> &args) { \
        param->args = &args; \
        InferShape(*inShape, outShape); \
        param->inputShapes = inShape; \
        param->outShape = outShape; \
        CREATE_OPERATOR(context, param, name, { \
            if (memory) { \
                memorySize = sizeof(DType) * param->outShape->Size(); \
            } else { \
                memorySize = 0; \
            }\
        }) \
    }\

namespace matrix {

    class Operator {
    public:

        Operator() {

        }

        virtual ~Operator() {
            input->clear();
            inputShapes->clear();
        }

        inline bool HasArg(const std::string &name) {
            return args->count(name) > 0;
        }

        template <class T>
        inline T GetArgValue(const std::string & name, const T &default_value) {
            if (args->count(name)) {
                return get<T>(args->at(name));
            }
            return default_value;
        }

        template <class T>
        inline T GetArgValue(const std::string & name) {
            if (args->count(name)) {
                return get<T>(args->at(name));
            }

            Logger::Global()->Fatal("can not find arg name");
            T t;
            return t;
        }


        template <class T>
        inline const T* Input(int idx) const {
            return static_cast<T*>(input->at(idx));
        }

        template <class T>
        inline T* InputNonConst(int idx) {
            return static_cast<T*>(input->at(idx));
        }

        template <class T>
        inline T* Output() {
            return static_cast<T*>(output);
        }


        inline void FallThrow() {
            output = input->at(0);
        }

        virtual bool Run() {
            return false;
        }

        virtual void AsyncRun() {

        }

        inline const int InputSize() const {
            return input->size();
        }

        inline void SetData(std::vector<void *> *input, void *output) {
            this->input = input;
            this->output = output;
        }

        virtual void VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        }

        virtual bool RunOnDevice() = 0;


    protected:
        std::map<std::string, Any> *args{nullptr};
        std::vector<void*> *input {nullptr};
        void* output{nullptr};
        std::vector<Shape*> *inputShapes{nullptr};
        Shape* outputShape{nullptr};
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

        std::vector<Shape*> *inputShapes{nullptr};
        Shape* outShape{nullptr};
        std::map<std::string, Any> *args{nullptr};
    };



    class OperatorProperty {
    public:
        OperatorProperty() = default;
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape)  = 0;
        virtual Operator* CreateOperator(Context context,
                                         std::vector<Shape *> *inShape, Shape *outShape,
                                         std::map<std::string, Any> &args) {
            return nullptr;
        }


        void SwitchType(const MatrixType &type) {
            this->param->type = type;
        }

        long GetMemorySize() {
            return memorySize;
        }

    protected:
        long memorySize = 0;
        Parameter * param {nullptr};
    };

}


#endif //MATRIX_OPERATOR_H
