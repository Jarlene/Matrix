//
// Created by Jarlene on 2018/2/8.
//

#ifndef MATRIX_BASEML_H
#define MATRIX_BASEML_H



namespace matrix {


    class BaseMl {
    public:
        virtual void Train() = 0;

        virtual void Classify() = 0;

    };

}
#endif //MATRIX_BASEML_H
