//
// Created by Jarlene on 2017/7/24.
//

#ifndef MATRIX_LOGGING_H
#define MATRIX_LOGGING_H

#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace matrix {

    enum class LogLevel : int {
        Debug = 0,
        Info = 1,
        Error = 2,
        Fatal = 3
    };


    class Logger {
    public:

        static Logger* Global() ;

        static Logger* Global(const std::string &name);

        explicit Logger(LogLevel level = LogLevel::Info);

        explicit Logger(std::string filename, LogLevel level = LogLevel::Info);

        ~Logger();

        int ResetLogFile(std::string filenam);

        int ResetLogLevel(LogLevel level);

        void ResetKillFatal(bool isKill);

        void Write(LogLevel level, const char* format, ...);
        void Info(const char* format, ...);
        void Debug(const char* format, ...);
        void Error(const char* format, ...);
        void Fatal(const char* format, ...);

        template<class T>
        void static PrintMat(const T *mat, int x, int y, std::string comment = "unknown") {
            int index = 0;
            std::cout << std::endl << comment << std::endl;
            for (int i = 0; i < x; ++i) {
                for (int j = 0; j < y; ++j) {
                    std::cout << std::setw(6) << mat[index++] << "  ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    private:
        void WriteImpl(LogLevel level, const char* format, va_list* val);
        void CloseLogFile();
        std::string GetSysTime();
        std::string GetLevelStr(LogLevel level);
        std::mutex mutex;
        std::FILE* file;
        LogLevel  level;
        bool isKill;
        Logger(const Logger&);
        void operator=(const Logger&);
    };


}

#endif //MATRIX_LOGGING_H
