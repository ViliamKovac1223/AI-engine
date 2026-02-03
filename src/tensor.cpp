#include "tensor.hpp"

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
            os << tensor.data[i] << (i != tensor.totalSize - 1 ? " " : "");
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
        os << tensor.data[i + *usedData] << (i != tensor.shape.back() - 1 ? " " : "");
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

Tensor Tensor::operator+(const Tensor& other) {
    return this->tensorsOperations(other, [](double a, double b) {return a + b;});
}

Tensor Tensor::operator*(const Tensor& other) {
    return this->tensorsOperations(other, [](double a, double b) {return a * b;});
}

Tensor Tensor::operator-(const Tensor& other) {
    return *this + (-1.0 * other);
}

Tensor Tensor::operator/(const Tensor& other) {
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
