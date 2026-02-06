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
    /**
     * Constructor for Tensor
     * @param shape Defines shape (dimensions) of the new tensor
     * @param defaultValue Value that tensor's elements will be initialized with
     */
    Tensor(const std::vector<size_t>& shape, float defaultValue);

    /**
     * Constructor for Tensor. Fills elements with randomly generated values.
     * @param shape Defines shape (dimensions) of the new tensor
     */
    Tensor(const std::vector<size_t>& shape);

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    // Math operations with other Tensors
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    /**
     * Computes mulmat operation at tensors
     * @param other Second tensor for mulmat operation
     * @return Result of mulmat as tensor
     */
    Tensor mulmat(const Tensor& other) const;

    // Math operations with numbers
    friend Tensor operator+(const double number, const Tensor& other);
    friend Tensor operator+(const Tensor& other, const double number);
    friend Tensor operator-(const double number, const Tensor& other);
    friend Tensor operator-(const Tensor& other, const double number);
    friend Tensor operator*(const double number, const Tensor& other);
    friend Tensor operator*(const Tensor& other, const double number);
    friend Tensor operator/(const double number, const Tensor& other);
    friend Tensor operator/(const Tensor& other, const double number);

    /**
     * Define seed for random generation. This seed will be used for every new
     * Tensor with randomly generated values.
     * @param Seed for random generation
     */
    static void seed(uint64_t seed);

private:
    /**
     * @return Random number between 0 and 1
     */
    static double getRandomNumber();

    /**
     * Recursively travel through Tensors dimensions, and print its data to os
     * @param os Stream to print data to
     * @param tensor Tensor to traverse
     * @param startingDim Traversing dimension
     * @param usedData Pointer to index of data that was already printed (in manual calling set to 0)
     * @return The given stream from parameters
     */
    static std::ostream& toStreamHelper(std::ostream& os, const Tensor& tensor,
        size_t startingDim, size_t * usedData);

    /**
     * Compares two tensors' shapes
     * @return Return true if the tensors' shapes are the same
     */
    bool compareShape(const Tensor& other) const;

    /**
     * Perform math operation on Tensors.
     * Example usage:
     this->tensorsOperations(other, [](double a, double b) {return a + b;});
     * @param other Second tensor to do math operation on
     * @param operation Function that will be called at every element from
     * tensors, result will be stored in the result tensor
     * @return Returns result tensor
     */
    Tensor tensorsOperations(const Tensor& other,
        std::function<double(double, double)> operation) const;

    /**
     * Perform math operation on Tensor and number.
     * Example usage:
     this->tensorsOperations(number, [](double a, double b) {return a + b;});
     * @param number Second number to do math operation on. This number will be 'b' in example above
     * @param operation Function that will be called at every element from
     * tensor and number, result will be stored in the result tensor
     * @return Returns result tensor
     */
    Tensor tensorsOperations(const double number,
        std::function<double(double, double)> operation) const;

    /**
     * Recursively calculates mulmat on tensors, and saves result to the res.
     * @param other Other tensor for calculation
     * @param res Result tensor
     * @param shapeIndexes Indicates which data in batch dimensions were already processed
     * (in manual call resize to the this->shape.size())
     * @param dim Current dimension (in manual call set to 0)
     */
    void mulmat(const Tensor& other, Tensor& res,
        std::vector<size_t>& shapeIndexes, size_t dim) const;

    /**
     * Calculate offset for t.data, based on already processed indexes (shapeIndexes)
     * @param shapeIndexes Indicates which data in batch dimensions were already processed
     * @param t Processed tensor
     */
    static size_t getMemoryOffset(std::vector<size_t> shapeIndexes, const Tensor& t);
};

#endif
