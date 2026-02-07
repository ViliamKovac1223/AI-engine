#include "tensor.hpp"
#include <vector>

Tensor::Tensor(const std::vector<size_t>& shape, float defaultValue)
    :shape(shape)
{
    std::cout << "tensor with default value" << std::endl;
    // Calculate memory size from shape
    this->totalSize = shape[0];
    for (size_t i = 1; i < shape.size(); i++) {
        this->totalSize *= shape[i];
    }

    // Allocate memory and initialize it
    this->data = std::make_unique<double[]>(this->totalSize);
    for (size_t i = 0; i < this->totalSize; i++)
        this->data[i] = defaultValue;
}

Tensor::Tensor(const std::vector<size_t>& shape)
    :shape(shape)
{
    // Calculate memory size from shape
    this->totalSize = shape[0];
    for (size_t i = 1; i < shape.size(); i++) {
        this->totalSize *= shape[i];
    }

    // Allocate memory
    this->data = std::make_unique<double[]>(this->totalSize);
    // initialize the memory with random values
    for (size_t i = 0; i < this->totalSize; i++)
        this->data[i] = getRandomNumber();
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    // Print shape
    os << tensor.shape.size() << "-D Tensor: [";
    for (size_t i = 0; i < tensor.shape.size(); i++) {
        const size_t dim = tensor.shape[i];
        os << dim << (i != tensor.shape.size() - 1 ? " " : "");
    }
    os << "]" << std::endl;

    // Print 1D tensor
    if (tensor.shape.size() == 1) {
        os << "[";
        for (size_t i = 0; i < tensor.totalSize; ++i) {
            os << tensor.data[i] << (i != tensor.totalSize - 1 ? ", " : "");
        }
        os << "]";
        return os;
    }

    // Print multi-dimensional tensor
    size_t usedData = 0;
    os << "[";
    Tensor::toStreamHelper(os, tensor, 0, &usedData);
    os << "]";
    return os;
}

std::ostream& Tensor::toStreamHelper(std::ostream& os, const Tensor& tensor, size_t startingDim, size_t * usedData) {
    if (tensor.shape.size() == 0) return os;

    // Loop to the last dimension, where data will be printed
    if (tensor.shape.size() - startingDim != 1) {
        for (size_t i = 0; i < tensor.shape[startingDim]; i++) {
            os << "[";
            Tensor::toStreamHelper(os, tensor, startingDim + 1, usedData);
            os << (i != tensor.shape[startingDim] - 1 ? "],\n" : "]");
        }
        return os;
    }

    // Print actuall data
    for (size_t i = 0; i < tensor.shape.back(); i++) {
        os << tensor.data[i + *usedData] << (i != tensor.shape.back() - 1 ? ", " : "");
    }
    *usedData += tensor.shape.back();

    return os;
}

bool Tensor::compareShape(const Tensor& other) const {
    if (other.shape.size() != this->shape.size()) return false;

    // Check if all dimensions are the same
    for (size_t i = 0; i < this->shape.size(); i++)
        if (other.shape[i] != this->shape[i])
            return false;

    return true;
}

void Tensor::seed(uint64_t seed) {
    gen.seed(seed);
}

double Tensor::getRandomNumber() {
    return dis(gen);
}

Tensor Tensor::mulmat(const Tensor& other) const {
    if (this->shape.size() == 0 || other.shape.size() != this->shape.size())
        return Tensor({0}, 0);

    // Calculate mulmat for 1D tensor (just do dot product)
    if (this->shape.size() == 1) {
        double finalProduct = 0.0;
        for (size_t i = 0; i < this->shape[0]; i++) {
            finalProduct += this->data[i] * other.data[i];
        }
        return Tensor({1}, finalProduct);
    }

    // Make shape for result
    std::vector<size_t> resShape(this->shape);
    resShape[this->shape.size() - 2] = this->shape[this->shape.size() - 2];
    resShape[this->shape.size() - 1] = other.shape[this->shape.size() - 1];
    Tensor result(resShape, 0);

    // Make empty shapeIndexes
    // This will serve as marker
    // To mark currently calculated batch dimension
    std::vector<size_t> shapeIndexes;
    shapeIndexes.resize(this->shape.size());

    // Perform mulmat
    mulmat(other, result, shapeIndexes, 0);
    return result;
}

void Tensor::mulmat(const Tensor& other, Tensor& res, std::vector<size_t>& shapeIndexes, size_t dim) const {
    // Go deeper in batch dimensions till we get to last 2 dims so we can
    // perform matrix mulmat
    if (this->shape.size() - 2 != dim) {
        for (size_t i = 0; i < this->shape[dim]; i++) {
            shapeIndexes[dim] = i;
            mulmat(other, res, shapeIndexes, dim + 1);
        }
        return;
    }

    // Calculate the base offset in memory to get the data we need (last two
    // dimensions, offset by specific indexes in previous dimensions)
    size_t baseIndex = getMemoryOffset(shapeIndexes, *this);
    size_t otherBaseIndex = getMemoryOffset(shapeIndexes, other);
    size_t resBaseIndex = getMemoryOffset(shapeIndexes, res);

    // Calculate mulmat
    size_t rows = this->shape[this->shape.size() - 2];
    size_t cols = this->shape[this->shape.size() - 1];
    size_t otherCols = other.shape[other.shape.size() - 1];
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < otherCols; j++) {
            for (size_t k = 0; k < cols; k++) {
                res.data[resBaseIndex + i * otherCols + j] +=
                    this->data[baseIndex + i * cols + k] *
                    other.data[otherBaseIndex + k * otherCols + j];
            }
        }
    }
}

size_t Tensor::getMemoryOffset(std::vector<size_t> shapeIndexes, const Tensor& t) {
    size_t baseIndex = 0;
    for (size_t i = 0; i < shapeIndexes.size() - 2; i++) {
        const size_t dim = shapeIndexes[i];
        size_t nextSizes = 1;
        for (size_t j = i + 1; j < t.shape.size(); j++) {
            nextSizes *= t.shape[j];
        }
        baseIndex += dim * nextSizes;
    }
    return baseIndex;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return this->tensorsOperations(other, [](double a, double b) {return a + b;});
}

Tensor Tensor::operator*(const Tensor& other) const {
    return this->tensorsOperations(other, [](double a, double b) {return a * b;});
}

Tensor Tensor::operator-(const Tensor& other) const {
    return *this + (-1.0 * other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    Tensor factor = 1.0 / other;
    return *this * factor;
}

Tensor operator+(const double number, const Tensor& other) {
    return other.tensorsOperations(number, [](double a, double b) {return a + b;});
}

Tensor operator+(const Tensor& other, const double number) {
    return number + other;
}

Tensor operator-(const double number, const Tensor& other) {
    return other.tensorsOperations(number, [](double a, double b) {return b - a;});
}

Tensor operator-(const Tensor& other, const double number) {
    return other.tensorsOperations(number, [](double a, double b) {return a - b;});
}

Tensor operator*(const double number, const Tensor& other) {
    return other.tensorsOperations(number, [](double a, double b) {return a * b;});
}

Tensor operator*(const Tensor& other, const double number) {
    return other.tensorsOperations(number, [](double a, double b) {return a * b;});
}

Tensor operator/(const double number, const Tensor& other) {
    return other.tensorsOperations(number, [](double a, double b) {return b / a;});
}

Tensor operator/(const Tensor& other, const double number) {
    return other.tensorsOperations(number, [](double a, double b) {return a / b;});
}

Tensor Tensor::tensorsOperations(const Tensor& other, std::function<double(double, double)> operation) const {
    Tensor result(this->shape);
    if (this->compareShape(other) == false)
        return result;

    for (size_t i = 0; i < this->totalSize; i++)
        result.data[i] = operation(this->data[i], other.data[i]);

    return result;
}

Tensor Tensor::tensorsOperations(const double number, std::function<double(double, double)> operation) const {
    Tensor result(this->shape);

    for (size_t i = 0; i < this->totalSize; i++)
        result.data[i] = operation(this->data[i], number);

    return result;
}

Tensor Tensor::mean() const {
    Tensor result({1}, 0);

    // Calculate mean and save it to result tensor
    double sum = 0;
    for (size_t i = 0; i < this->totalSize; i++)
        sum += this->data[i];
    result.data[0] = sum / this->totalSize;

    return result;
}

Tensor Tensor::max() const {
    Tensor result({1}, 0);

    // Find max and save it to the result tensor
    double max = this->data[0];
    for (size_t i = 1; i < this->totalSize; i++)
        if (this->data[i] > max)
            max = this->data[i];
    result.data[0] = max;

    return result;
}

Tensor Tensor::min() const {
    Tensor result({1}, 0);

    // Find min and save it to the result tensor
    double min = this->data[0];
    for (size_t i = 1; i < this->totalSize; i++)
        if (this->data[i] < min)
            min = this->data[i];
    result.data[0] = min;

    return result;
}

Tensor Tensor::sum() const {
    Tensor result({1}, 0);

    // Calculate sum and save it to result tensor
    double sum = 0;
    for (size_t i = 0; i < this->totalSize; i++)
        sum += this->data[i];
    result.data[0] = sum;

    return result;
}
