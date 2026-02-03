#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstdint>
#include <ostream>
#include <random>
#include <iostream>
#include <memory>
#include <vector>
#include <ostream>
#include <functional>

class Tensor {
private:
    std::unique_ptr<double[]> data;
    std::vector<size_t> shape;
    size_t totalSize;

    static inline std::mt19937 gen;
    static inline std::uniform_real_distribution<double> dis{0.0, 1.0};

public:
    Tensor(const std::vector<size_t>& shape, float defaultValue);
    Tensor(const std::vector<size_t>& shape);

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    // Math operations with other Tensors
    Tensor operator+(const Tensor& other);
    Tensor operator-(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor operator/(const Tensor& other);

    // Math operations with numbers
    friend Tensor operator+(const double number, const Tensor& other);
    friend Tensor operator+(const Tensor& other, const double number);
    friend Tensor operator-(const double number, const Tensor& other);
    friend Tensor operator-(const Tensor& other, const double number);
    friend Tensor operator*(const double number, const Tensor& other);
    friend Tensor operator*(const Tensor& other, const double number);
    friend Tensor operator/(const double number, const Tensor& other);
    friend Tensor operator/(const Tensor& other, const double number);

    static void seed(uint64_t seed);

private:
    static double getRandomNumber();
    static std::ostream& toStreamHelper(std::ostream& os, const Tensor& tensor,
        size_t startingDim, size_t * usedData);

    bool compareShape(const Tensor& other) const;
    Tensor tensorsOperations(const Tensor& other, std::function<double(double, double)> operation) const;
    Tensor tensorsOperations(const double number, std::function<double(double, double)> operation) const;
};

#endif
