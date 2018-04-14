//
// Created by Jarlene on 2018/4/14.
//

#include <matrix/include/utils/Dispatcher.h>

#include <iostream>

using namespace std;
using namespace matrix;

int print(const std::string &name) {
    cout << "the params is "<< name << endl;
//    throw std::runtime_error(name);
    return 1;
}

void success(int res) {
    cout << "the success param "<< res << endl;
}

void fail(std::exception &e) {
    cout << "fail: " << e.what() << endl;
}

void complete() {
    cout << "complete " << endl;
}

int main() {
    auto x = dispatch(print, "sss");
    cout << x << endl;
    dispatch(print, success, fail, complete, "hahndes");

}