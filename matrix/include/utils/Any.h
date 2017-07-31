//
// Created by Jarlene on 2017/7/21.
//

#ifndef MATRIX_ANY_H
#define MATRIX_ANY_H


#include <typeinfo>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <assert.h>


namespace matrix {

    class  Any;

    template<typename T>
    inline T& get(Any& src);

    template<typename T>
    inline const T& get(const Any& src);

    class Any {
    public:
        inline Any() = default;

        inline Any(Any&& other);

        inline Any(const Any& other);

        template<typename T>
        inline Any(T&& other);

        inline ~Any();

        inline Any& operator=(Any&& other);

        inline Any& operator=(const Any& other);

        template<typename T>
        inline Any& operator=(T&& other);

        inline bool empty() const;

        inline void clear();

        inline void swap(Any& other);

        inline const std::type_info& type() const;

    private:
        template<typename T>
        class TypeOnHeap;
        template<typename T>
        class TypeOnStack;
        template<typename T>
        class TypeInfo;

        static const size_t kStack = sizeof(void*) * 3;
        static const size_t kAlign = sizeof(void*);

        union Data {
            // stack space
            std::aligned_storage<kStack, kAlign>::type stack;
            // pointer to heap space
            void* pheap;
        };

        struct Type {
            // destructor function
            void (*destroy)(Data* data);
            // copy constructor
            void (*create_from_data)(Data* dst, const Data& src);
            // the type info function
            const std::type_info* ptype_info;
        };

        template<typename T>
        struct data_on_stack {
            static const bool value = alignof(T) <= kAlign && sizeof(T) <= kStack;
        };

        template<typename T>
        friend T& get(Any& src);

        template<typename T>
        friend const T& get(const Any& src);

        inline void construct(Any&& other);

        inline void construct(const Any& other);

        template<typename T>
        inline void check_type() const;
        // internal type specific information
        const Type* type_{nullptr};
        // internal data
        Data data_;
    };


    template<typename T>
    class Any::TypeOnHeap {
    public:
        inline static T* get_ptr(Any::Data* data) {
            return static_cast<T*>(data->pheap);
        }
        inline static const T* get_ptr(const Any::Data* data) {
            return static_cast<const T*>(data->pheap);
        }
        inline static void create_from_data(Any::Data* dst, const Any::Data& data) {
            dst->pheap = new T(*get_ptr(&data));
        }
        inline static void destroy(Data* data) {
            delete static_cast<T*>(data->pheap);
        }
    };

    template<typename T>
    class Any::TypeOnStack {
    public:
        inline static T* get_ptr(Any::Data* data) {
            return reinterpret_cast<T*>(&(data->stack));
        }
        inline static const T* get_ptr(const Any::Data* data) {
            return reinterpret_cast<const T*>(&(data->stack));
        }
        inline static void create_from_data(Any::Data* dst, const Any::Data& data) {
            new (&(dst->stack)) T(*get_ptr(&data));
        }
        inline static void destroy(Data* data) {
            T* dptr = reinterpret_cast<T*>(&(data->stack));
            dptr->~T();
        }
    };

    template<typename T>
    class Any::TypeInfo : public std::conditional<Any::data_on_stack<T>::value, Any::TypeOnStack<T>, Any::TypeOnHeap<T> >::type {
    public:
        inline static const Type* get_type() {
            static TypeInfo<T> tp;
            return &(tp.type_);
        }

    private:
        // local type
        Type type_;
        // constructor
        TypeInfo() {
            if (std::is_pod<T>::value) {
                type_.destroy = nullptr;
            } else {
                type_.destroy = TypeInfo<T>::destroy;
            }
            type_.create_from_data = TypeInfo<T>::create_from_data;
            type_.ptype_info = &typeid(T);
        }
    };




    Any::Any(Any &&other) {
        this->construct(other);
    }

    Any::Any(const Any &other) {
        this->construct(other);
    }

    template<typename T>
    Any::Any(T &&other) {
        typedef typename std::decay<T>::type DT;
        if (std::is_same<DT, Any>::value) {
            this->construct(std::forward<T>(other));
        } else {
            static_assert(std::is_copy_constructible<DT>::value,
                          "Any can only hold value that is copy constructable");
            type_ = TypeInfo<DT>::get_type();
            if (data_on_stack<DT>::value) {
                new (&(data_.stack)) DT(std::forward<T>(other));
            } else {
                data_.pheap = new DT(std::forward<T>(other));
            }
        }
    }

    Any::~Any() {
        this->clear();
    }

    Any &Any::operator=(Any &&other) {
        Any(std::move(other)).swap(*this);
        return *this;
    }

    Any &Any::operator=(const Any &other) {
        Any(other).swap(*this);
        return *this;
    }

    template<typename T>
    Any &Any::operator=(T &&other) {
        Any(std::forward<T>(other)).swap(*this);
        return *this;
    }

    bool Any::empty() const {
        return type_ == nullptr;
    }

    void Any::clear() {
        if (type_ != nullptr) {
            if (type_->destroy != nullptr) {
                type_->destroy(&data_);
            }
            type_ = nullptr;
        }
    }

    void Any::swap(Any &other) {
        std::swap(type_, other.type_);
        std::swap(data_, other.data_);
    }

    const std::type_info &Any::type() const {
        if (type_ != nullptr) {
            return *(type_->ptype_info);
        } else {
            return typeid(void);
        }
    }

    template<typename T>
    T &get(Any &src) {
        src.check_type<T>();
        return *Any::TypeInfo<T>::get_ptr(&(src.data_));
    }

    template<typename T>
    const T &get(const Any &src) {
        src.check_type<T>();
        return *Any::TypeInfo<T>::get_ptr(&(src.data_));
    }

    void Any::construct(Any &&other) {
        type_ = other.type_;
        data_ = other.data_;
        other.type_ = nullptr;
    }

    void Any::construct(const Any &other) {
        type_ = other.type_;
        if (type_ != nullptr) {
            type_->create_from_data(&data_, other.data_);
        }
    }

    template<typename T>
    void Any::check_type() const {
        assert(type_ != nullptr);
        assert(type_->ptype_info == &typeid(T));
    }

}


#endif //MATRIX_ANY_H
