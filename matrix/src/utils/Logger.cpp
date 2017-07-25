//
// Created by Jarlene on 2017/7/24.
//

#include "matrix/include/utils/Logger.h"

namespace matrix {

    Logger::Logger(LogLevel level) {
        this->level = level;
        this->file = nullptr;
        isKill = true;
    }

    Logger::Logger(std::string filename, LogLevel level) {
        this->level = level;
        this->file = nullptr;
        ResetLogFile(filename);
    }

    Logger::~Logger() {
        CloseLogFile();
    }

    int Logger::ResetLogFile(std::string filename) {
        CloseLogFile();
        if (filename.size() > 0) {
#ifdef _MSC_VER
            fopen_s(&file, filename.c_str(), "w");
#else
            file = fopen(filename.c_str(), "w");
#endif
            if (file == nullptr) {
                Write(LogLevel::Error, "Cannot create log file %s\n", filename.c_str());
                return -1;
            }
        }

        return 0;
    }

    int Logger::ResetLogLevel(LogLevel level) {
        this->level = level;
        return 0;
    }

    void Logger::ResetKillFatal(bool isKill) {
        this->isKill = isKill;
    }

    void Logger::Write(LogLevel level, const char *format, ...) {
        va_list val;
        va_start(val, format);
        WriteImpl(level, format, &val);
        va_end(val);
    }

    void Logger::Info( const char *format, ...) {
        va_list val;
        va_start(val, format);
        WriteImpl(LogLevel::Info, format, &val);
        va_end(val);
    }

    void Logger::Debug(const char *format, ...) {
        va_list val;
        va_start(val, format);
        WriteImpl(LogLevel::Debug, format, &val);
        va_end(val);
    }

    void Logger::Error( const char *format, ...) {
        va_list val;
        va_start(val, format);
        WriteImpl(LogLevel::Error, format, &val);
        va_end(val);
    }

    void Logger::Fatal(const char *format, ...) {
        va_list val;
        va_start(val, format);
        WriteImpl(LogLevel::Fatal, format, &val);
        va_end(val);
    }

    void Logger::WriteImpl(LogLevel level, const char *format, va_list *val) {
        if (level >= this->level) {  // omit the message with low level

            std::string level_str = GetLevelStr(level);

            std::string time_str = GetSysTime();

            va_list val_copy;

            va_copy(val_copy, *val);

            // write to STDOUT

            printf("[%s] [%s] ", level_str.c_str(), time_str.c_str());

            vprintf(format, *val);

            fflush(stdout);

            // write to log file

            if (file != nullptr) {

                fprintf(file, "[%s] [%s] ", level_str.c_str(), time_str.c_str());

                vfprintf(file, format, val_copy);

                fflush(file);

            }

            va_end(val_copy);



            if (isKill && level == LogLevel::Fatal) {
                CloseLogFile();
                exit(1);
            }

        }
    }

    void Logger::CloseLogFile() {
        if (file != nullptr) {
            fclose(file);
            file = nullptr;
        }
    }

    std::string Logger::GetSysTime() {
        time_t t = time(NULL);
        char str[64];

#ifdef _MSC_VER

        tm time;
        localtime_s(&time, &t);
        strftime(str, sizeof(str), "%Y-%m-%d %H:%M:%S", &time);
#else
        strftime(str, sizeof(str), "%Y-%m-%d %H:%M:%S", localtime(&t));
#endif

        return str;
    }

    std::string Logger::GetLevelStr(LogLevel level) {
        switch (level) {
            case LogLevel::Debug: return "DEBUG";
            case LogLevel::Info: return "INFO";
            case LogLevel::Error: return "ERROR";
            case LogLevel::Fatal: return "FATAL";
            default: return "UNKNOW";
        }
    }

    Logger *Logger::Global(std::string &name) {
        static Logger logger(name);
        logger.ResetKillFatal(true);
        return &logger;
    }

    Logger *Logger::Global() {
        static Logger logger;
        logger.ResetKillFatal(true);
        return &logger;
    }


}
