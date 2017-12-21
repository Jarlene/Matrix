//
// Created by Jarlene on 2017/12/18.
//

#ifndef MATRIX_EMBEDDING_H
#define MATRIX_EMBEDDING_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <matrix/include/utils/Logger.h>

using namespace std;

vector<string> split(const string &s, const string &seperator) {
    vector<string> result;
    typedef string::size_type string_size;
    string_size i = 0;

    while (i != s.size()) {
        //找到字符串中首个不等于分隔符的字母；
        int flag = 0;
        while (i != s.size() && flag == 0) {
            flag = 1;
            for (string_size x = 0; x < seperator.size(); ++x)
                if (s[i] == seperator[x]) {
                    ++i;
                    flag = 0;
                    break;
                }
        }

        //找到又一个分隔符，将两个分隔符之间的字符串取出；
        flag = 0;
        string_size j = i;
        while (j != s.size() && flag == 0) {
            for (string_size x = 0; x < seperator.size(); ++x)
                if (s[j] == seperator[x]) {
                    flag = 1;
                    break;
                }
            if (flag == 0)
                ++j;
        }
        if (i != j) {
            result.push_back(s.substr(i, j - i));
            i = j;
        }
    }
    return result;
}


class Embedding {
public:
    Embedding(const string &train, const string &test, const int embed) {
        ifstream trainFs(train.data());
        ifstream testFs(test);
        if (!trainFs || !testFs) {
            std::cerr << " can not find train file path or test file path" << endl;
        }
        vector<vector<string>> testData;
        vector<vector<string>> trainData;
        set<string> lookuptable;
        string line;
        while (!trainFs.eof()) {
            getline(trainFs, line);
            auto res = split(line, "*&");
            if (res.size() != 2) {
                continue;
            }
            this->trainLabel.push_back(atof(res[1].data()));
            auto list = split(res[0], " ");
            trainData.push_back(list);
            for (auto s : list) {
                if (s == "\t" || s == "") {
                    continue;
                }
                lookuptable.insert(s);
            }
        }
        trainFs.close();

        while (!testFs.eof()) {
            getline(testFs, line);
            auto res = split(line, "*&");
            if (res.size() != 2) {
                continue;
            }
            this->testLabel.push_back(atof(res[1].data()));
            auto list = split(res[0], " ");
            testData.push_back(list);
            for (auto s : list) {
                if (s == "\t" || s == "") {
                    continue;
                }
                lookuptable.insert(s);
            }
        }
        testFs.close();

        std::normal_distribution<float> dist_normal(0.1f, 1.0f);
        gettimeofday(&tv, NULL);
        rnd_engine_.seed((unsigned int) (tv.tv_sec * 1000 * 1000 + tv.tv_usec));
        for (auto s : lookuptable) {
            float *data = static_cast<float *>(malloc(sizeof(float) * embed));
            for (int i = 0; i < embed; ++i) {
                *data++ = dist_normal(rnd_engine_);
            }
            matrix::Logger::PrintMat(data, 1, embed, s);
            lookup_table[s] = data;
        }

        for (auto it : trainData) {
            vector<float *> temp;
            for (auto subid : it) {
                temp.push_back(lookup_table[subid]);
            }
            this->trainData.push_back(temp);
        }

        for (auto it : testData) {
            vector<float *> temp;
            for (auto subid : it) {
                temp.push_back(lookup_table[subid]);
            }
            this->testData.push_back(temp);
        }

        this->indexTable = new int[this->trainData.size()];
        for (int i = 0; i < this->trainData.size(); ++i) {
            this->indexTable[i] = i;
        }
    }

    ~Embedding() {
        for (auto it : lookup_table) {
            free(it.second);
        }
        lookup_table.clear();
        for (auto it : trainData) {
            it.clear();
        }
        trainData.clear();
        for (auto it : testData) {
            it.clear();
        }
        testData.clear();
        if (indexTable != nullptr) {
            free(indexTable);
            indexTable = nullptr;
        }
    }


    void getTrainBatch(int batch, float *data, float *label) {
        for (int i = 0; i < batch; ++i) {
            label[i] = this->trainLabel[this->indexTable[this->currentBatchPos]];
            int idx = 0;
            for (auto it : this->trainData[this->indexTable[this->currentBatchPos]]) {
                *data++ = it[idx];
                idx++;
            }
            this->currentBatchPos++;
            if (this->currentBatchPos >= trainData.size()) {
                this->shuffle();
                this->currentBatchPos = 0;
            }
        }
    }


private:
    int shuffle() {
        std::default_random_engine random(time(NULL));
        for (int i = 0; i < this->trainData.size(); ++i) {
            std::uniform_int_distribution<int> dis1(i, this->trainData.size());
            int rand_pos = dis1(random);
            int tmp = this->indexTable[i];
            this->indexTable[i] = this->indexTable[rand_pos];
            this->indexTable[rand_pos] = tmp;
        }
        return 0;
    }


private:
    struct timeval tv;
    std::mt19937 rnd_engine_;
    int currentBatchPos;
    int *indexTable;
    map<string, float *> lookup_table;
    vector<vector<float *>> trainData;
    vector<float> trainLabel;
    vector<vector<float *>> testData;
    vector<float> testLabel;

};


#endif //MATRIX_EMBEDDING_H
