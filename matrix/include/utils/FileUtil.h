//
// Created by Jarlene on 2018/1/26.
//

#ifndef MATRIX_FILEUTIL_H
#define MATRIX_FILEUTIL_H

#include <cassert>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <ftw.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <functional>

using namespace std;


namespace matrix {


    enum FileSizeUnit {
        BYTE = 0,
        KB,
        MB,
        GB,
        TB,
        PB
    };


    static void iterate_files_worker(const string &path,
                                     std::function<void(const string &file, bool is_dir)> func,
                                     bool recurse);

    static string getFileName(const string &path) {
        string rc = path;
        auto pos = path.find_last_of('/');
        if (pos != string::npos) {
            rc = path.substr(pos + 1);
        }
        return rc;
    }

    static string getFileExt(const string &path) {
        string rc = getFileName(path);
        auto pos = rc.find_last_of('.');
        if (pos != string::npos) {
            rc = rc.substr(pos);
        } else {
            rc = "";
        }
        return rc;
    }


    static string join(const string &s1, const string s2) {
        string rc;
        if (s2.size() > 0) {
            if (s2[0] == '/') {
                rc = s2;
            } else if (s1.size() > 0) {
                rc = s1;
                if (rc[rc.size() - 1] != '/') {
                    rc += "/";
                }
                rc += s2;
            } else {
                rc = s2;
            }
        } else {
            rc = s1;
        }
        return rc;
    }

    static string getFileParentDir(const string &path) {
        string rc = path;
        auto pos = path.find_last_of('/');
        if (pos != string::npos) {
            rc = path.substr(0, pos);
        }
        return rc;
    }

    static double getFileSize(const string &path, FileSizeUnit unit = KB) {
        struct stat stats;
        if (stat(path.c_str(), &stats) == -1) {
            throw std::runtime_error("Could not find file: \"" + path + "\"");
        }
        double base = stats.st_size * 1.0;
        switch (unit) {
            case BYTE:
                return base;
            case KB:
                return base/1024;
            case MB:
                return base/1024/1024;
            case GB:
                return base/1024/1024/1024;
            case TB:
                return base/1024/1024/1024/1024;
            case PB:
                return base/1024/1024/1024/1024/1024;
        }
    }


    static void iterate_files(const string &path,
                              std::function<void(const string &file, bool is_dir)> func,
                              bool recurse) {
        vector<string> files;
        vector<string> dirs;
        iterate_files_worker(path,
                             [&files, &dirs](const string &file, bool is_dir) {
                                 if (is_dir)
                                     dirs.push_back(file);
                                 else
                                     files.push_back(file);
                             },
                             recurse);

        for (auto f : files) {
            func(f, false);
        }
        for (auto f : dirs) {
            func(f, true);
        }
    }


    static void iterate_files_worker(const string &path,
                                     std::function<void(const string &file, bool is_dir)> func,
                                     bool recurse) {
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(path.c_str())) != nullptr) {
            while ((ent = readdir(dir)) != nullptr) {
                string name = ent->d_name;
                switch (ent->d_type) {
                    case DT_DIR:
                        if (name != "." && name != "..") {
                            string dir_path = join(path, name);
                            if (recurse) {
                                iterate_files(dir_path, func, recurse);
                            }
                            func(dir_path, true);
                        }
                        break;
                    case DT_LNK:
                        break;
                    case DT_REG: {
                        string file_name = join(path, name);
                        func(file_name, false);
                        break;
                    }
                    default:
                        break;
                }
            }
            closedir(dir);
        } else {
            throw std::runtime_error("error enumerating file " + path);
        }
    }

    static int removeDir(const string &dir) {
        struct stat status;
        if (stat(dir.c_str(), &status) != -1) {
            iterate_files(dir,
                          [](const string &file, bool is_dir) {
                              if (is_dir) {
                                  rmdir(file.c_str());
                              } else {
                                  remove(file.c_str());
                              }
                          },
                          true);
            rmdir(dir.c_str());
            return 0;
        }
        return 1;
    }

    static int removeFile(const string &file) {
        remove(file.c_str());
        return 0;
    }


    static bool makeDir(const string &dir) {
        if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
            if (errno == EEXIST) {
                return false;
            }
            throw std::runtime_error("error making directory " + dir + " " + strerror(errno));
        }
        return true;
    }


    static vector<char> read(const string &path) {
        size_t file_size = getFileSize(path);
        vector<char> data;
        data.reserve(file_size);
        data.resize(file_size);

        FILE *f = fopen(path.c_str(), "rb");
        if (f) {
            char *p = data.data();
            size_t remainder = file_size;
            size_t offset = 0;
            while (f && remainder > 0) {
                size_t rc = fread(&p[offset], 1, remainder, f);
                offset += rc;
                remainder -= rc;
            }
            fclose(f);
        } else {
            throw std::runtime_error("error opening file '" + path + "'");
        }
        return data;
    }

    static std::string read_file_to_string(const std::string &path) {
        std::ifstream f(path);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }


    static bool exists(const std::string &filename) {
        struct stat buffer;
        return (stat(filename.c_str(), &buffer) == 0);
    }

    static int try_get_lock(const std::string &filename) {
        mode_t m = umask(0);
        int fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
        umask(m);
        if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) < 0) {
            close(fd);
            fd = -1;
        }
        return fd;
    }

    static void release_lock(int fd, const std::string &filename) {
        if (fd >= 0) {
            removeFile(filename);
            close(fd);
        }
    }

    static time_t get_timestamp(const std::string &filename) {
        time_t rc = 0;
        struct stat st;
        if (stat(filename.c_str(), &st) == 0) {
            rc = st.st_mtime;
        }
        return rc;
    }
}


#endif //MATRIX_FILEUTIL_H
