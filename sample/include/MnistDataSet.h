//
// Created by Jarlene on 2017/12/8.
//

#ifndef MATRIX_MNISTDATASET_H
#define MATRIX_MNISTDATASET_H


class MnistDataSet {
public:
    MnistDataSet(string dataPath, string labelPath) {
        cout<<"MnistDataSet::MnistDataSet" <<endl;
        int numberOfImages;
        int imageSize;
        auto reverseInt = [](int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        };

        typedef unsigned char uchar;

        cout<< "read image" << endl;
        ifstream data_file(dataPath, ios::binary);

        if(data_file.is_open()) {
            int magic_number = 0, n_rows = 0, n_cols = 0;

            data_file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

            data_file.read((char *)&numberOfImages, sizeof(numberOfImages)); numberOfImages = reverseInt(numberOfImages);
            data_file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
            data_file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

            imageSize = n_rows * n_cols;
            this->data = new float*[numberOfImages];
            uchar* image = new uchar[imageSize];
            cout<< "numberOfImage=" << numberOfImages << endl;
            for (int i=0; i<numberOfImages; ++i) {
                this->data[i] = new float[imageSize];
                data_file.read((char*)&image[0], imageSize);
                for (int j=0; j<imageSize; ++j) {
                    this->data[i][j] = image[j]*1.0/256.0f;
                }
            }

            free(image);
            this->imageSize = n_rows * n_cols;
            this->numberOfImages = numberOfImages;
        } else {
            throw runtime_error("Cannot open file `" + dataPath + "`!");
        }

        cout<<"read label" <<endl;
        ifstream label_file(labelPath, ios::binary);
        if(label_file.is_open()) {
            int magic_number = 0;
            int number_of_labels;
            label_file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

            label_file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

            cout<<"number of label=" << number_of_labels << endl;
            uchar* _dataset = new uchar[number_of_labels];
            label_file.read((char*)&_dataset[0], number_of_labels);
            this->label = new float[number_of_labels];
            for (int i=0; i<number_of_labels; ++i) {
                this->label[i] = _dataset[i];
            }
            free(_dataset);
        } else {
            throw runtime_error("Unable to open file `" + labelPath + "`!");
        }
        this->currentBatchPos = 0;
        this->indexTable = new int[this->numberOfImages];
        for (int i=0; i<numberOfImages; ++i) {
            this->indexTable[i] = i;
        }
    }

    ~MnistDataSet() {
        for (int i=0; i<this->numberOfImages; ++i) {
            free(data[i]);
        }
        free(data);
        free(label);

        free(this->indexTable);
    }

    int getNumberOfImages() {return this->numberOfImages;}
    int getImageSize() { return this->imageSize; }
    float** getData() {return data;}
    float* getLabel() {return label;}

    int getMiniBatch(int miniBathchSize, float* _image, float* _label) {
        for (int i=0; i<miniBathchSize; ++i) {
            _label[i] = this->label[this->indexTable[this->currentBatchPos]];
            for (int j=0; j<this->imageSize; ++j) {
                *_image++ = this->data[this->indexTable[this->currentBatchPos]][j];
            }
            this->currentBatchPos++;
            if(this->currentBatchPos==this->numberOfImages) {
                this->shuffle();
                this->currentBatchPos=0;
            }
        }
        return 0;
    }

    int shuffle() {
        std::default_random_engine random(time(NULL));

        for (int i=0; i<this->numberOfImages; ++i) {
            std::uniform_int_distribution<int> dis1(i, this->numberOfImages);
            int rand_pos = dis1(random);

            int tmp = this->indexTable[i];
            this->indexTable[i] = this->indexTable[rand_pos];
            this->indexTable[rand_pos] = tmp;
        }

        return 0;
    }

private:
    float** data;
    float* label;
    int numberOfImages;
    int imageSize;
    int currentBatchPos;
    int* indexTable;
};

#endif //MATRIX_MNISTDATASET_H
