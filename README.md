# accessor-BLAS
The purpose of this repository is to both showcase the performance of the Ginkgo accessor and to serve as an integration example.

## Including the Ginkgo's accessor into your project

### Integration into the build system
To use Ginkgo's accessor, you need to:
1. Use C++14 or higher in your own project
2. use `${GINKGO_DIR}` as an include directory (you only need `${GINKGO_DIR}/accessor` from Ginkgo)

In this repository, we use CMake, which makes the integration straight forward.
We give users the option to either specify a local copy of the Ginkgo repository, or automatically clone the repository into the build directory, followed by using it.
We achieve both with these few lines in CMake:  
https://github.com/ginkgo-project/accessor-BLAS/blob/10adcedbbbdc67a5f1093ff7284890085d0d24e4/CMakeLists.txt#L16-L43  
Now, we only need to call this function for every target where we use the accessor.


### Creating a range with an accessor
In this repository, we only use the `reduced_row_major` accessor, but all others work accordingly.

For the `reduced_row_major` accessor, you need to specify:
1. the dimensionality of the range (we specify 2D, even for vectors, so we can access vectors with a stride)
2. the arithmetic and storage type
Now, this type can be used to create the `range<reduced_row_major<...>>` by specifying the size, stride and storage pointer.

We showcase the creation of both constant and non-constant ranges with `reduced_row_major` accessors here:  
https://github.com/ginkgo-project/accessor-BLAS/blob/10adcedbbbdc67a5f1093ff7284890085d0d24e4/cuda/gemv_kernels.cuh#L178-L189  
We utilize the constant accessors for the matrix and the x vector since both storage pointers are also constant.


### Utilizing the range/accessor in a CUDA kernel

Utilizing the range in a kernel (works the same for CPUs) is straight forward:
1. Use a templated kernel argument in order to accept all kind of ranges
2. Read and write operations utilize the bracket operator()

To know which arithmetic type is used, we can either use the `accessor::arithmetic_type` property, or detect what type arithmetic operations result in. In this example, we use the second option:  
https://github.com/ginkgo-project/accessor-BLAS/blob/10adcedbbbdc67a5f1093ff7284890085d0d24e4/cuda/gemv_kernels.cuh#L86

Read and write options can be observed in GEMV here:  
https://github.com/ginkgo-project/accessor-BLAS/blob/10adcedbbbdc67a5f1093ff7284890085d0d24e4/cuda/gemv_kernels.cuh#L110


### Difference between using plain pointers and using the range/accessor

Here, we compare the GEMV kernel written with plain pointers:  
https://github.com/ginkgo-project/accessor-BLAS/blob/10adcedbbbdc67a5f1093ff7284890085d0d24e4/cuda/gemv_kernels.cuh#L30-L64

and using the range/accessor:  
https://github.com/ginkgo-project/accessor-BLAS/blob/10adcedbbbdc67a5f1093ff7284890085d0d24e4/cuda/gemv_kernels.cuh#L79-L113


The main differences between these are:
1. We have fewer parameters in the range/accesser kernel because stride and size information are integrated into the ranges.
2. We don't need to compute a 1D index with the range/accessor because indices to both dimensions are fed into the brace operator
3. For debug purposes (and showcase), we extract the arithmetic type from the accessor.
