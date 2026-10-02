// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct SparseSoftmaxTilingData {
    uint32_t length;
    uint32_t mode;
    int32_t dim;

    uint32_t outerSize;
    uint32_t dimSize;
    uint32_t innerSize;

    uint32_t ptrLength;
    uint32_t indexLength;
    
    float eps;
};