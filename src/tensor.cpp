#include "tensor.hpp"
#include <functional>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

Tensor::Tensor(const std::vector<size_t>& shape, double defaultValue)
    :Tensor(shape, defaultValue, false)
{}

Tensor::Tensor(const std::vector<size_t>& shape, double defaultValue,
    bool requiresGrad, const std::string& operation,
    const std::unordered_set<Tensor, HashFunction>& children)
    :Tensor(shape, defaultValue, requiresGrad)
{
    this->operation = operation;
    this->prev = children;
}

Tensor::Tensor(const std::vector<size_t>& shape,
    bool requiresGrad, const std::string& operation,
    const std::unordered_set<Tensor, HashFunction>& children)
    :Tensor(shape, requiresGrad)
{
    this->operation = operation;
    this->prev = children;
}

Tensor::Tensor(const std::vector<size_t>& shape, double defaultValue, bool requiresGrad)
    :shape(shape)
{
    this->requiresGrad = requiresGrad;
    this->isGradInit = false;
    this->grad = nullptr;
    if (requiresGrad)
        this->grad = std::make_shared<Tensor>(shape, 0.0);
    this->_backward = nullptr;
    this->operation = "";

    // Calculate memory size from shape
    this->totalSize = shape[0];
    for (size_t i = 1; i < shape.size(); i++) {
        this->totalSize *= shape[i];
    }

    // Allocate memory and initialize it
    this->data = std::make_shared<double[]>(this->totalSize);
    for (size_t i = 0; i < this->totalSize; i++)
        this->data[i] = defaultValue;
}

Tensor::Tensor(const std::vector<size_t>& shape)
    :Tensor(shape, false)
{}

Tensor::Tensor(const std::vector<size_t>& shape, bool requiresGrad)
    :shape(shape)
{
    this->requiresGrad = requiresGrad;
    this->isGradInit = false;
    this->grad = nullptr;
    if (requiresGrad)
        this->grad = std::make_shared<Tensor>(shape, 0.0);
    this->_backward = nullptr;
    this->operation = "";

    // Calculate memory size from shape
    this->totalSize = shape[0];
    for (size_t i = 1; i < shape.size(); i++) {
        this->totalSize *= shape[i];
    }

    // Allocate memory
    this->data = std::make_shared<double[]>(this->totalSize);
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

Tensor Tensor::mulmat(Tensor& other) {
    if (this->shape.size() == 0 || other.shape.size() != this->shape.size())
        return Tensor({0}, 0.0);

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
    Tensor result(resShape, 0.0);
    bool requiresGrad = this->requiresGrad || other.requiresGrad;
    if (requiresGrad) {
        std::unordered_set<Tensor, HashFunction> children = {*this, other};
        result = Tensor(resShape, 0.0, requiresGrad, operation, children);
    }

    // Make empty shapeIndexes
    // This will serve as marker
    // To mark currently calculated batch dimension
    std::vector<size_t> shapeIndexes;
    shapeIndexes.resize(this->shape.size());

    // Perform mulmat
    mulmat(other, result, shapeIndexes, 0);
    return result;
}

void Tensor::mulmat(Tensor& other, Tensor& res, std::vector<size_t>& shapeIndexes, size_t dim) {
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

    if (!requiresGrad)
        return;

    // If no previous backward function is found, let it be empty function
    std::function<void()> prevBackFunc = []() {};
    // Get previous backward function to use later
    if (res._backward != nullptr)
        prevBackFunc = *res._backward;

    // Define backward function for backpropagation if needed
    Tensor& a = *this;
    res._backward = std::make_shared<std::function<void()>>(
        [prevBackFunc, &a, other, res, rows, cols, otherCols, baseIndex, resBaseIndex, otherBaseIndex]() {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < otherCols; j++) {
                    for (size_t k = 0; k < cols; k++) {
                        a.grad->data[baseIndex + i * cols + k] +=
                            res.grad->data[resBaseIndex + i * otherCols + j] *
                            other.data[otherBaseIndex + k * otherCols + j];
                        other.grad->data[otherBaseIndex + k * otherCols + j] +=
                            a.data[baseIndex + i * cols + k] *
                            res.grad->data[resBaseIndex + i * otherCols + j];
                    }
                }
            }
            a.grad->isGradInit = true;
            other.grad->isGradInit = true;

            // Execute previously defined backward function.
            // Because we go through batch dimensions recursively we have to
            // chain backward functions like this.
            prevBackFunc();
        });
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

bool Tensor::operator==(const Tensor& other) const {
    if (this->compareShape(other) == false) return false;

    for (size_t i = 0; this->totalSize; i++) {
        if (this->data[i] != other.data[i]) {
            return false;
        }
    }

    return true;
}

Tensor Tensor::operator+(Tensor& other) {
    return Tensor::tensorsOperations(*this, other, "+");
}

Tensor Tensor::operator*(Tensor& other) {
    return Tensor::tensorsOperations(*this, other, "*");
}

Tensor Tensor::operator-(Tensor& other) {
    Tensor tmp = -1.0 * other;
    return *this + tmp;
}

Tensor Tensor::operator/(Tensor& other) {
    Tensor factor = 1.0 / other;
    return *this * factor;
}

Tensor operator+(double number, Tensor& other) {
    return Tensor::tensorsOperations(other, number, "+");
}

Tensor operator+(Tensor& other, double number) {
    return number + other;
}

Tensor operator-(double number, Tensor& other) {
    return Tensor::tensorsOperations(number, other, "-");
}

Tensor operator-(Tensor& other, double number) {
    return Tensor::tensorsOperations(other, number, "-");
}

Tensor operator*(double number, Tensor& other) {
    return Tensor::tensorsOperations(other, number, "*");
}

Tensor operator*(Tensor& other, double number) {
    return Tensor::tensorsOperations(other, number, "*");
}

Tensor operator/(double number, Tensor& other) {
    return Tensor::tensorsOperations(number, other, "/");
}

Tensor operator/(Tensor& other, const double number) {
    return Tensor::tensorsOperations(other, number, "/");
}

Tensor Tensor::tensorsOperations(Tensor& a, Tensor& b, std::string operation) {
    Tensor result(a.shape, 0.0);
    bool requiresGrad = a.requiresGrad || b.requiresGrad;
    if (requiresGrad) {
        std::unordered_set<Tensor, HashFunction> children = {a, b};
        result = Tensor(a.shape, 0.0, requiresGrad, operation, children);
    }

    if (a.compareShape(b) == false)
        return result;

    std::unordered_map<std::string, std::function<double(double, double)>> operationMap{
        {"+", [](double a, double b) {return a + b;}},
        {"*", [](double a, double b) {return a * b;}},
    };

    // Do basic math operation between tensors
    if (operationMap.find(operation) != operationMap.end())
        for (size_t i = 0; i < a.totalSize; i++)
            result.data[i] = operationMap.at(operation)(a.data[i], b.data[i]);

    if (!requiresGrad)
        return result;

    // Define backward function for calculating gradient if needed
    result._backward = std::make_shared<std::function<void()>>([operationMap, operation, &result, a, b]() {
        for (size_t i = 0; i < result.grad->totalSize; i++) {
            if (operation == "*") {
                a.grad->data[i] += result.grad->data[i] * b.data[i];
                b.grad->data[i] += result.grad->data[i] * a.data[i];
            } else if (operation == "+") {
                a.grad->data[i] += result.grad->data[i];
                b.grad->data[i] += result.grad->data[i];
            }
        }
        a.grad->isGradInit = true;
        b.grad->isGradInit = true;
    });

    return result;
}

Tensor Tensor::tensorsOperations(Tensor& a, double number, std::string operation) {
    Tensor result(a.shape, 0.0);
    bool requiresGrad = a.requiresGrad;
    if (requiresGrad) {
        std::unordered_set<Tensor, HashFunction> children = {a};
        result = Tensor(a.shape, 0.0, requiresGrad, operation, children);
    }

    std::unordered_map<std::string, std::function<double(double, double)>> operationMap{
        {"+", [](double a, double b) {return a + b;}},
        {"-", [](double a, double b) {return a - b;}},
        {"*", [](double a, double b) {return a * b;}},
        {"/", [](double a, double b) {return a / b;}}
    };

    std::unordered_map<std::string, std::function<double(double, double)>> gradOpMap {
        {"+", [](double a, [[maybe_unused]] double b) {return a;}},
        {"-", [](double a, [[maybe_unused]] double b) {return a;}},
        {"*", [](double a, double b) {return a * b;}},
        {"/", [](double a, double b) {return a / b;}}
    };

    // Do basic math operations
    if (operationMap.find(operation) != operationMap.end())
        for (size_t i = 0; i < a.totalSize; i++)
            result.data[i] = operationMap.at(operation)(a.data[i], number);

    if (!requiresGrad)
        return result;

    // Define backward function for calculating gradient if needed
    result._backward = std::make_shared<std::function<void()>>([gradOpMap, operation, result, a, number]() {
        for (size_t i = 0; i < result.grad->totalSize; i++) {
            a.grad->data[i] += gradOpMap.at(operation)(result.grad->data[i], number);
        }
        a.grad->isGradInit = true;
    });

    return result;
}

Tensor Tensor::tensorsOperations(double number, Tensor& a, std::string operation) {
    Tensor result(a.shape, 0.0);
    bool requiresGrad = a.requiresGrad;
    if (requiresGrad) {
        std::unordered_set<Tensor, HashFunction> children = {a};
        result = Tensor(a.shape, 0.0, requiresGrad, operation, children);
    }

    std::unordered_map<std::string, std::function<double(double, double)>> operationMap{
        {"+", [](double a, double b) {return a + b;}},
        {"-", [](double a, double b) {return a - b;}},
        {"*", [](double a, double b) {return a * b;}},
        {"/", [](double a, double b) {return a / b;}}
    };

    std::unordered_map<std::string, std::function<double(double, double, double)>> gradOpMap {
        {"+", [](double a, [[maybe_unused]] double b, [[maybe_unused]] double aData)
            {return a;}},
        {"-", [](double a, [[maybe_unused]] double b, [[maybe_unused]] double aData)
            {return -a;}},
        {"*", [](double a, double b, [[maybe_unused]] double aData)
            {return a * b;}},
        {"/", [](double a, double b, double aData)
            {return a * (-b / (aData * aData));}}
    };

    // Do basic math operations
    if (operationMap.find(operation) != operationMap.end())
        for (size_t i = 0; i < a.totalSize; i++)
            result.data[i] = operationMap.at(operation)(number, a.data[i]);

    if (!requiresGrad)
        return result;

    // Define backward function for calculating gradient if needed
    result._backward = std::make_shared<std::function<void()>>([gradOpMap, operation, &result, a, number]() {
        for (size_t i = 0; i < result.grad->totalSize; i++) {
            a.grad->data[i] += gradOpMap.at(operation)(result.grad->data[i], number, a.data[i]);
        }
        a.grad->isGradInit = true;
    });

    return result;
}

Tensor Tensor::mean() {
    Tensor result({1}, 0.0);
    bool requiresGrad = this->requiresGrad;
    if (requiresGrad) {
        std::unordered_set<Tensor, HashFunction> children = {*this};
        result = Tensor({1}, 0.0, requiresGrad, operation, children);
    }

    // Calculate mean and save it to result tensor
    double sum = 0;
    for (size_t i = 0; i < this->totalSize; i++)
        sum += this->data[i];
    result.data[0] = sum / this->totalSize;

    if (!requiresGrad)
        return result;

    // Add backward function for backward propagation
    Tensor& a = *this;
    result._backward = std::make_shared<std::function<void()>>([&a, &result]() {
        double n = (double) a.totalSize;

        for (size_t i = 0; i < a.grad->totalSize; i++) {
            a.grad->data[i] += result.grad->data[0] * (1.0 / n);
        }
        a.grad->isGradInit = true;
    });

    return result;
}

Tensor Tensor::max() {
    Tensor result({1}, 0.0);
    bool requiresGrad = this->requiresGrad;
    if (requiresGrad) {
        std::unordered_set<Tensor, HashFunction> children = {*this};
        result = Tensor({1}, 0.0, requiresGrad, operation, children);
    }

    // Find max and save it to the result tensor
    double max = this->data[0];
    size_t maxIndex = 0;
    for (size_t i = 1; i < this->totalSize; i++) {
        if (this->data[i] > max) {
            max = this->data[i];
            maxIndex = i;
        }
    }
    result.data[0] = max;

    if (!requiresGrad)
        return result;

    // Add backward function for backward propagation
    Tensor& a = *this;
    result._backward = std::make_shared<std::function<void()>>([&a, &result, maxIndex]() {
        // Only max element gets gradient update
        a.grad->data[maxIndex] += result.grad->data[0];
        a.grad->isGradInit = true;
    });

    return result;
}

Tensor Tensor::min() {
    Tensor result({1}, 0.0);
    bool requiresGrad = this->requiresGrad;
    if (requiresGrad) {
        std::unordered_set<Tensor, HashFunction> children = {*this};
        result = Tensor({1}, 0.0, requiresGrad, operation, children);
    }

    // Find min and save it to the result tensor
    double min = this->data[0];
    size_t minIndex = 0;
    for (size_t i = 1; i < this->totalSize; i++) {
        if (this->data[i] < min) {
            min = this->data[i];
            minIndex = i;
        }
    }
    result.data[0] = min;

    if (!requiresGrad)
        return result;

    // Add backward function for backward propagation
    Tensor& a = *this;
    result._backward = std::make_shared<std::function<void()>>([&a, &result, minIndex]() {
        // Only min element gets gradient update
        a.grad->data[minIndex] += result.grad->data[0];
        a.grad->isGradInit = true;
    });

    return result;
}

Tensor Tensor::sum() {
    Tensor result({1}, 0.0);
    bool requiresGrad = this->requiresGrad;
    if (requiresGrad) {
        std::unordered_set<Tensor, HashFunction> children = {*this};
        result = Tensor({1}, 0.0, requiresGrad, operation, children);
    }

    // Calculate sum and save it to result tensor
    double sum = 0;
    for (size_t i = 0; i < this->totalSize; i++)
        sum += this->data[i];
    result.data[0] = sum;

    if (!requiresGrad)
        return result;

    // Add backward function for backward propagation
    Tensor& a = *this;
    result._backward = std::make_shared<std::function<void()>>([&a, &result]() {
        double n = (double) a.totalSize;

        for (size_t i = 0; i < a.grad->totalSize; i++) {
            a.grad->data[i] += result.grad->data[0];
        }
        a.grad->isGradInit = true;
    });

    return result;
}

void Tensor::backward() {
    std::vector<const Tensor*> topo;
    std::vector<const Tensor*> visited;

    // Find all visited nodes and put all unique nodes to topo
    std::function<void(const Tensor*)> build_topo = [&](const Tensor* v) {
        // Check if already visited this node
        if (std::find(visited.begin(), visited.end(), v) == visited.end()) {
            visited.push_back(v);
            for (const Tensor& p : v->prev) {
                const Tensor * p1 = &p;
                build_topo(p1);
            }
            topo.push_back(v);
        }
    };
    build_topo(this);

    // Set first gradient to 1.0
    *this->grad = Tensor((std::vector<size_t>) {1}, (double) 1.0);
    this->grad->isGradInit = true;

    // Process nodes in reverse order
    for (auto it = topo.rbegin(); it != topo.rend(); it++) {
        if ((*it)->_backward != nullptr)
            (*(*it)->_backward)();
    }
}
