#pragma once

#include <type_traits>


namespace detail {


template <typename ValueType, typename = void>
struct atomic_helper {
    __forceinline__ __device__ static ValueType atomic_add(ValueType *,
                                                           ValueType)
    {
        static_assert(sizeof(ValueType) == 0,
                      "This default function is not implemented, only the "
                      "specializations are.");
    }
    __forceinline__ __device__ static ValueType atomic_max(ValueType *,
                                                           ValueType)
    {
        static_assert(sizeof(ValueType) == 0,
                      "This default function is not implemented, only the "
                      "specializations are.");
    }
};

template <typename ResultType, typename ValueType>
__forceinline__ __device__ ResultType reinterpret(ValueType val)
{
    static_assert(sizeof(ValueType) == sizeof(ResultType),
                  "The type to reinterpret to must be of the same size as the "
                  "original type.");
    return reinterpret_cast<ResultType &>(val);
}

#define BIND_ATOMIC_HELPER_STRUCTURE(CONVERTER_TYPE)                         \
    template <typename ValueType>                                            \
    struct atomic_helper<                                                    \
        ValueType,                                                           \
        std::enable_if_t<(sizeof(ValueType) == sizeof(CONVERTER_TYPE))>> {   \
        __forceinline__ __device__ static ValueType atomic_add(              \
            ValueType *__restrict__ addr, ValueType val)                     \
        {                                                                    \
            using c_type = CONVERTER_TYPE;                                   \
            return atomic_wrapper(addr, [&val](c_type &old, c_type assumed,  \
                                               c_type *c_addr) {             \
                old = atomicCAS(c_addr, assumed,                             \
                                reinterpret<c_type>(                         \
                                    val + reinterpret<ValueType>(assumed))); \
            });                                                              \
        }                                                                    \
                                                                             \
    private:                                                                 \
        template <typename Callable>                                         \
        __forceinline__ __device__ static ValueType atomic_wrapper(          \
            ValueType *__restrict__ addr, Callable set_old)                  \
        {                                                                    \
            CONVERTER_TYPE *address_as_converter =                           \
                reinterpret_cast<CONVERTER_TYPE *>(addr);                    \
            CONVERTER_TYPE old = *address_as_converter;                      \
            CONVERTER_TYPE assumed;                                          \
            do {                                                             \
                assumed = old;                                               \
                set_old(old, assumed, address_as_converter);                 \
            } while (assumed != old);                                        \
            return reinterpret<ValueType>(old);                              \
        }                                                                    \
    };

// Support 64-bit ATOMIC_ADD
BIND_ATOMIC_HELPER_STRUCTURE(unsigned long long int);
// Support 32-bit ATOMIC_ADD
BIND_ATOMIC_HELPER_STRUCTURE(unsigned int);


}  // namespace detail


// Default implementation using the atomic_helper structure
template <typename T>
__forceinline__ __device__ T atomic_add(T *__restrict__ addr, T val)
{
    return detail::atomic_helper<T>::atomic_add(addr, val);
}

// Specific implementation using the CUDA function directly
#define BIND_ATOMIC_ADD(ValueType)                   \
    __forceinline__ __device__ ValueType atomic_add( \
        ValueType *__restrict__ addr, ValueType val) \
    {                                                \
        return atomicAdd(addr, val);                 \
    }

BIND_ATOMIC_ADD(int);
BIND_ATOMIC_ADD(unsigned int);
BIND_ATOMIC_ADD(unsigned long long int);
BIND_ATOMIC_ADD(float);

#if !((defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) || \
      (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)))
// CUDA 8.0 starts suppoting 64-bit double atomicAdd on devices of compute
// capability 6.x and higher
BIND_ATOMIC_ADD(double);
#endif  // !((defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) ||
        // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)))
