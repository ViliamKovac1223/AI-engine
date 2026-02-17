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

## Example Code
This project includes [example of simple linear model](example/). This code
showcases simple linear model for predicting rent prices. Dataset is stored in
[dataset.hpp](example/include/dataset.hpp) for simplicity, the original data is
from
[archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified),
and this data is licensed at the original dataset license.

The example code starts with normalizing the data (scale data between 0.0 and
1.0), and splitting them between training and evaluation data. After training
loop we evaluate this model on evaluation data. Also seed is set for
reproducibility.

Results of this model are:
- Initial MSE loss (before training) starts at ``1.54212``
- MSE loss on training data after training is ``0.0128``
- MSE loss on evaluation data after training is ``0.062``

To get this example code running, don't forget to obtain a header-only
version of this library and put it in include folder, more details in [section above](##example-code).
Then just use included Makefile.
