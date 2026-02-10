#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstdint>
#include <ostream>
#include <random>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>
#include <ostream>
#include <functional>

class Tensor {
private:
    struct HashFunction {
        // Calculate hash from tensor
        size_t operator()(const Tensor& tensor) const {
            size_t dataHash = 0;
            for (size_t i = 0; i < tensor.totalSize; i++) {
                dataHash ^= std::hash<double>()(tensor.data[i]);
            }
            size_t shapeHash = 0;
            for (size_t i = 0; i < tensor.shape.size(); i++) {
                shapeHash ^= std::hash<size_t>()(tensor.shape[i]);
            }

            return dataHash ^ shapeHash;
        }
    };

    std::shared_ptr<double[]> data;
    std::vector<size_t> shape;
    size_t totalSize;

    bool requiresGrad;
    bool isGradInit;
    mutable std::shared_ptr<std::function<void()>> _backward;
    mutable std::unordered_set<Tensor, HashFunction> prev;
    mutable std::string operation;

    static inline std::mt19937 gen;
    static inline std::uniform_real_distribution<double> dis{0.0, 1.0};

public:
    mutable std::shared_ptr<Tensor> grad;

    /**
     * Constructor for Tensor
     * @param shape Defines shape (dimensions) of the new tensor
     * @param defaultValue Value that tensor's elements will be initialized with
     */
    Tensor(const std::vector<size_t>& shape, double defaultValue);

    /**
     * Constructor for Tensor
     * @param shape Defines shape (dimensions) of the new tensor
     * @param defaultValue Value that tensor's elements will be initialized with
     * @param requiresGrad set if gradient is required for this tensor
     */
    Tensor(const std::vector<size_t>& shape, double defaultValue, bool requiresGrad);

    /**
     * Constructor for Tensor. Fills elements with randomly generated values.
     * @param shape Defines shape (dimensions) of the new tensor
     */
    Tensor(const std::vector<size_t>& shape);

    /**
     * Constructor for Tensor. Fills elements with randomly generated values.
     * @param shape Defines shape (dimensions) of the new tensor
     * @param requiresGrad set if gradient is required for this tensor
     */
    Tensor(const std::vector<size_t>& shape, bool requiresGrad);

    /**
     * Constructor for Tensor. Fills elements with randomly generated values.
     * @param shape Defines shape (dimensions) of the new tensor
     * @param requiresGrad set if gradient is required for this tensor
     * @param operation string operation that was used in creation of this tensor
     * @param children Nodes that this node was created from
     */
    Tensor(const std::vector<size_t>& shape,
        bool requiresGrad, const std::string& operation,
        const std::unordered_set<Tensor, HashFunction>& children);

    /**
     * Constructor for Tensor
     * @param shape Defines shape (dimensions) of the new tensor
     * @param defaultValue Value that tensor's elements will be initialized with
     * @param requiresGrad set if gradient is required for this tensor
     * @param operation string operation that was used in creation of this tensor
     * @param children Nodes that this node was created from
     */
    Tensor(const std::vector<size_t>& shape, double defaultValue,
        bool requiresGrad, const std::string& operation,
        const std::unordered_set<Tensor, HashFunction>& children);

    /**
     * Do backward propagation from this node to all its children nodes.
     * This works only on scalar tensors.
     */
    void backward();

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    // Math operations with other Tensors
    Tensor operator+(Tensor& other);
    Tensor operator-(Tensor& other);
    Tensor operator*(Tensor& other);
    Tensor operator/(Tensor& other);
    bool operator==(const Tensor& other) const;

    /**
     * Computes mulmat operation at tensors
     * @param other Second tensor for mulmat operation
     * @return Result of mulmat as tensor
     */
    Tensor mulmat(Tensor& other);

    /**
     * @return Mean from Tensor's values
     */
    Tensor mean();

    /**
     * @return Max value from Tensor
     */
    Tensor max();

    /**
     * @return Min value from Tensor
     */
    Tensor min();

    /**
     * @return Sum from Tensor's values
     */
    Tensor sum();

    // Math operations with numbers
    friend Tensor operator+(double number, Tensor& other);
    friend Tensor operator+(Tensor& other, double number);
    friend Tensor operator-(double number, Tensor& other);
    friend Tensor operator-(Tensor& other, double number);
    friend Tensor operator*(double number, Tensor& other);
    friend Tensor operator*(Tensor& other, double number);
    friend Tensor operator/(double number, Tensor& other);
    friend Tensor operator/(Tensor& other, double number);

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
     * @param a First tensor to do math operation on
     * @param b Second tensor to do math operation on
     * @param operation String representing the operation. Supported values "+",
     * "-", "/", "*", if unsported value is given then returned result is tensor
     * filled with zeroes
     * @return Returns result tensor
     */
    static Tensor tensorsOperations(Tensor& a, Tensor& b, std::string operation);

    /**
     * Perform math operation on Tensor and number.
     * @param a First tensor to do math operation on
     * @param number Second number to do math operation on.
     * @param operation String representing the operation. Supported values "+",
     * "-", "/", "*", if unsported value is given then returned result is tensor
     * filled with zeroes
     * @return Returns result tensor
     */
    static Tensor tensorsOperations(Tensor& a, double number, std::string operation);

    /**
     * Perform math operation on Tensor and number.
     * @param number First number to do math operation on.
     * @param a Second tensor to do math operation on
     * @param operation String representing the operation. Supported values "+",
     * "-", "/", "*", if unsported value is given then returned result is tensor
     * filled with zeroes
     * @return Returns result tensor
     */
    static Tensor tensorsOperations(double number, Tensor& a, std::string operation);

    /**
     * Recursively calculates mulmat on tensors, and saves result to the res.
     * @param other Other tensor for calculation
     * @param res Result tensor
     * @param shapeIndexes Indicates which data in batch dimensions were already processed
     * (in manual call resize to the this->shape.size())
     * @param dim Current dimension (in manual call set to 0)
     */
    void mulmat(Tensor& other, Tensor& res,
        std::vector<size_t>& shapeIndexes, size_t dim);

    /**
     * Calculate offset for t.data, based on already processed indexes (shapeIndexes)
     * @param shapeIndexes Indicates which data in batch dimensions were already processed
     * @param t Processed tensor
     */
    static size_t getMemoryOffset(std::vector<size_t> shapeIndexes, const Tensor& t);
};

#endif
