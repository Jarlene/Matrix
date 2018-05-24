//
// Created by Jarlene on 2018/4/16.
//

#ifndef MATRIX_STRINGUTIL_H
#define MATRIX_STRINGUTIL_H

#include <string>
#include <sstream>

using namespace std;


namespace matrix {

    template <class T>
    const string toString(T &val) {
        return std::to_string(val);
    }


    template <class T>
    const T fromString(string &val) {
        static_assert(std::is_arithmetic<T>::value, "T must is number");
        istringstream is(val);
        T t(0);
        is >> t;
        return t;
    }

    bool startWith(const string &src, const string &prefix) {
        if (src.size() < prefix.size()) return false;
        for (size_t i = 0; i < prefix.size(); i++) {
            if (src[i] != prefix[i]) return false;
        }
        return true;

    }


    bool endWith(const string &src, const string &suffix) {
        if (src.size() < suffix.size()) return false;
        size_t off = src.size() - suffix.size();
        for (size_t i = 0; i < suffix.size(); i++) {
            if (src[off + i] != suffix[i]) return false;
        }
        return true;
    }


    string replace(const string &str, const string &find, const string &replace, const int c = -1) {
        size_t pos = 0;
        string result = str;
        int count = 0;
        while ((pos = result.find(find, pos)) != string::npos) {
            result.replace(pos, find.length(), replace);
            pos += replace.length();
            count++;
            if (c != -1 && count >= c) {
                break;
            }
        }
        return result;
    }

    vector<string> split(const std::string &source, const std::string &sep) {
        vector<string> elements;
        size_t start = 0;
        size_t found = 0;
        while ((found = source.find(sep, start)) != std::string::npos) {
            elements.push_back(source.substr(start, found - start));
            start = found + sep.size();
        }

        if (start <= source.size()) {
            elements.push_back(source.substr(start, std::string::npos));
        }
        return elements;
    }



    string getEnv(const char * name) {
        if (!name) return "";
        char *lvl = getenv(name);
        if (lvl) return string(lvl);
        return "";
    }

}


#endif //MATRIX_STRINGUTIL_H
