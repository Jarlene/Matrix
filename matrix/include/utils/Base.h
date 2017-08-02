//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_BASE_H
#define MATRIX_BASE_H


#define INSTANCE_CLASS_TYPE(classname, type) \
template class matrix::classname<type>;

#define INSTANCE_CLASS(classname)  \
INSTANCE_CLASS_TYPE(classname, int) \
INSTANCE_CLASS_TYPE(classname, float) \
INSTANCE_CLASS_TYPE(classname, double)\
INSTANCE_CLASS_TYPE(classname, long)\

#endif //MATRIX_BASE_H
