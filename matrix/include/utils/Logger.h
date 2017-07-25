//
// Created by 郑珊 on 2017/7/24.
//

#ifndef MATRIX_LOGGING_H
#define MATRIX_LOGGING_H

#include <string>
#include <fstream>

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

        static Logger* Global(std::string &name);

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

    private:
        void WriteImpl(LogLevel level, const char* format, va_list* val);
        void CloseLogFile();
        std::string GetSysTime();
        std::string GetLevelStr(LogLevel level);

        std::FILE* file;
        LogLevel  level;
        bool isKill;
        Logger(const Logger&);
        void operator=(const Logger&);
    };


}

#endif //MATRIX_LOGGING_H
