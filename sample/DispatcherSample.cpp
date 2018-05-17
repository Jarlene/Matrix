//
// Created by Jarlene on 2018/4/14.
//

#include <matrix/include/utils/Dispatcher.h>
#include <matrix/include/utils//Logger.h>
#include <iostream>

using namespace std;
using namespace matrix;

int print(const std::string &name) {

    cout << "the params is "<< name << endl;
//    throw std::runtime_error(name);
    return 1;
}

void success(int res) {
    MLOG(INFO) << "MLOG the success param " << res ;
}

void fail(std::exception &e) {
    cout << "fail: " << e.what() << endl;
}

void complete() {
    cout << "complete " << endl;
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    dispatch(print, success, fail, complete, "hahndes");

}