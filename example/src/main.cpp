#include "tensor.hpp"
#include "dataset.hpp"
#include <vector>

std::vector<Tensor> getTensorData(std::vector<HouseData>& data);
void normalizeTensor(const std::vector<HouseData>& refrence, Tensor& X, Tensor& y);

const size_t DIM_IN = 3;
const size_t DIM_OUT = 1;

int main() {
    // Set seed for reproducibility
    Tensor::seed(42);

    // Get data
    auto xyData = getTensorData(data);
    Tensor X = xyData[0];
    Tensor y = xyData[1];
    Tensor Xe = xyData[2];
    Tensor ye = xyData[3];

    // Nomalize data
    normalizeTensor(data, X, y);
    normalizeTensor(data, Xe, ye);

    Tensor W({DIM_IN, DIM_OUT}, true);
    Tensor b({1}, true);

    double learning_rate = 0.005;
    size_t epochs = 15000;
    size_t logFreq = 1000;

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        // Calculate prediction
        Tensor yHat = X.mulmat(W) + b;

        // Calculate MSE loss
        Tensor loss = ((y - yHat).pow(2)).mean();

        loss.backward();

        // Update weights
        Tensor t = learning_rate * *W.grad;
        W = W - t;
        t = learning_rate * *b.grad;
        b = b - t;

        W.resetGrad();
        b.resetGrad();

        // Logs
        if (epoch % logFreq == 0 || epoch == epochs - 1) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss << std::endl;
        }
    }

    // Evaluate
    Tensor yHatEval = Xe.mulmat(W) + b;
    Tensor lossEval = ((ye - yHatEval).pow(2)).mean();
    std::cout << "eval loss: " << lossEval << std::endl;
}

void normalizeTensor(const std::vector<HouseData>& refrence, Tensor& X, Tensor& y) {
    size_t rowSize = DIM_OUT + DIM_IN;
    for (size_t i = 0; i < rowSize; i++) {
        size_t * refPtr = (size_t*) &(refrence[0]);
        double min = *(refPtr + i);
        double max = *(refPtr + i);

        // Find max and min
        for (size_t j = 1; j < refrence.size(); j++) {
            size_t * refPtr = (size_t*) &(refrence[j]);
            double val = *(refPtr + i);

            if (val > max) {
                max = val;
            } else if (val < min) {
                min = val;
            }
        }

        if (i >= rowSize - DIM_OUT) {
            for (size_t j = 0; j < y.getShape()[0]; j++)
                y[y.getShape()[1] * j + i - DIM_IN] = (y[y.getShape()[1] * j + i - DIM_IN] - min) / (max - min);
        } else {
            for (size_t j = 0; j < X.getShape()[0]; j++)
                X[X.getShape()[1] * j + i] = (X[X.getShape()[1] * j + i] - min) / (max - min);
        }
    }
}

std::vector<Tensor> getTensorData(std::vector<HouseData>& data) {
    const size_t totalSize = data.size();
    size_t trainingSize = totalSize * 0.8;
    size_t evalSize = totalSize - trainingSize;

    Tensor X({trainingSize, DIM_IN}, true);
    Tensor y({trainingSize, DIM_OUT}, true);

    Tensor Xe({evalSize, DIM_IN}, true);
    Tensor ye({evalSize, DIM_OUT}, true);

    // Shuffle original data before redustribution
    unsigned seed = 42;
    std::mt19937 g(seed);
    for (size_t i = 0; i < 500; i++)
        std::shuffle(data.begin(), data.end(), g);

    for (size_t i = 0; i < totalSize; i++) {
        if (i < trainingSize) {
            X[DIM_IN * i + 0] = data[i].size;
            X[DIM_IN * i + 1] = data[i].city;
            X[DIM_IN * i + 2] = data[i].state;
            y[i] = data[i].price;
        } else {
            Xe[DIM_IN * i + 0 - trainingSize] = data[i].size;
            Xe[DIM_IN * i + 1 - trainingSize] = data[i].city;
            Xe[DIM_IN * i + 2 - trainingSize] = data[i].state;
            ye[i - trainingSize] = data[i].price;
        }
    }

    return {X, y, Xe, ye};
}
