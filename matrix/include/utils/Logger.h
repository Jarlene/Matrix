//
// Created by Jarlene on 2017/7/24.
//

#ifndef MATRIX_LOGGING_H
#define MATRIX_LOGGING_H

#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>

#define RED "\x1b[31;1m"
#define GREEN "\x1b[32;1m"
#define CYAN "\x1b[36;1m"
#define WHITE "\x1b[37;1m"
#define BOLD "\x1b[0;1m"
#define RESET "\x1b[0m"

namespace matrix {

    enum class LogLevel : int {
        INFO = 0,
        WARNING = 1,
        ERROR = 2,
        FATAL = 3
    };


    class Logger {
    public:

        static Logger* Global() ;

        static Logger* Global(const std::string &name);

        explicit Logger(LogLevel level = LogLevel::INFO);

        explicit Logger(std::string filename, LogLevel level = LogLevel::INFO);

        ~Logger();

        int ResetLogFile(std::string filenam);

        int ResetLogLevel(LogLevel level);

        void ResetKillFatal(bool isKill);

        void Write(LogLevel level, const char* format, ...);
        void Info(const char* format, ...);
        void Debug(const char* format, ...);
        void Error(const char* format, ...);
        void Fatal(const char* format, ...);

        std::ostream& stream();

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

#define LOG(level) matrix::Logger(matrix::LogLevel::level).stream()<< "[" << __FILE__<< ":" <<  __LINE__ <<"("<< __FUNCTION__<<")" << "] \n    " << std::flush



#endif //MATRIX_LOGGING_H
