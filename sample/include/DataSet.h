//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_DATASET_H
#define MATRIX_DATASET_H

#include <string>
#include <vector>
#include <fstream>


using namespace std;

class DataSet {
public:

    DataSet(const string &dataPath, const string &labelPath) {
        read_Mnist_Data(dataPath, MniData);
        read_Mnist_Label(labelPath, MniLabel);
        currentBatchIndx = 0;
        allDataSize = MniData.size();
        oneDataSize = MniData[0].size();
    }

    void Reset() {
        currentBatchIndx = 0;
    }

    bool GetBatchData(int batchSize, float* data, float* label) {
        if (data == nullptr || label == nullptr) {
            cout<< "the input is null ptr " << endl;
        }
        if (batchSize == -1) {
            for (int i = 0; i < allDataSize; ++i) {
                memcpy(data + i * oneDataSize, MniData[currentBatchIndx + i].data(), oneDataSize * sizeof(float));
            }
            memcpy(label, MniLabel.data(), allDataSize * sizeof(float));
            return false;
        }

        if (currentBatchIndx >= MniData.size()){
            return false;
        }

        for (int i = 0; i < batchSize; ++i) {
            memcpy(data + i * oneDataSize, MniData[currentBatchIndx + i].data(), oneDataSize * sizeof(float));
        }
        memcpy(label, MniLabel.data(), batchSize * sizeof(float));
        currentBatchIndx += batchSize;
        return true;
    }

private:

    int reverseInt4MNIST(const int i) {
        unsigned char  ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }

    void read_Mnist_Data(string filename, vector<vector<float> > &vec) {
        ifstream file (filename, ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read((char*) &magic_number, sizeof(magic_number));
            magic_number = reverseInt4MNIST(magic_number);
            file.read((char*) &number_of_images,sizeof(number_of_images));
            number_of_images = reverseInt4MNIST(number_of_images);
            file.read((char*) &n_rows, sizeof(n_rows));
            n_rows = reverseInt4MNIST(n_rows);
            file.read((char*) &n_cols, sizeof(n_cols));
            n_cols = reverseInt4MNIST(n_cols);
            for(int i = 0; i < number_of_images; ++i) {
                vector<float> tp;
                for(int r = 0; r < n_rows; ++r) {
                    for(int c = 0; c < n_cols; ++c) {
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.push_back(temp * 1.0f / 255 - 0.5f);
                    }
                }
                vec.push_back(tp);
            }
        }
    }

    void read_Mnist_Label(string filename, vector<float> &vec) {
        ifstream file (filename, ios::binary);
        if (file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read((char*) &magic_number, sizeof(magic_number));
            magic_number = reverseInt4MNIST(magic_number);
            file.read((char*) &number_of_images,sizeof(number_of_images));
            number_of_images = reverseInt4MNIST(number_of_images);
            for(int i = 0; i < number_of_images; ++i)
            {
                unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
                vec.push_back((float)temp);
            }
        }
    }

private:
    int currentBatchIndx;
    long allDataSize;
    vector<vector<float>> MniData;
    vector<float> MniLabel;
    int oneDataSize;
};

#endif //MATRIX_DATASET_H
