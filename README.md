# AI Engine
This is a very small personal and educational project that provides basic
functionality to create AI models. AI models are made from tensors, and for AI
model to converge we need math operations for those tensors, and
backpropagation functionality, which is what this library provides. This is not
meant to be used in production systems, and it is only for educational
purposes.

## Math Operations and Backpropagation 
Supported math operations are:

- Addition
- Subtraction
- Multiplication
- Division
- Mulmat (matrix/tensor multiplication)
- Exponentiation (pow)
- Mean (returns scalar)
- Max (returns scalar)
- Min (returns scalar)
- Sum (returns scalar)
- Backpropagation (backward function)

## Use of library
To use this library in project we recommend to use this as header-only library.
To get this single header, you can compile it with following commands.

```
cd tools/
go build -o generator generator.go file_type.go
cd ..
./tools/generator
```

Generator supports following options (flags):
- ``--root <path_to_library>`` This configures a path to the library source files. Default path is ``.``.
- ``--out <path_for_header>`` This configures a path where single header file will be generated. Default path is ``./example/include/tensor.hpp``.
- ``--impl-macro <const_header_guard>`` This configures what header guard will be used before implementation inside of a header. Default is ``TENSOR_LIB_IMPL``.
