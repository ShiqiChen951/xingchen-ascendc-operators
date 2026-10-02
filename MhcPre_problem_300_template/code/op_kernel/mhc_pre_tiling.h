// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcPreTilingData {
    uint32_t length;
    // x shape = (B, S, n, D)
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t n;
    uint32_t hiddenDim;

    // 派生维度
    uint32_t rowNum;       // B * S
    uint32_t nD;           // n * D
    uint32_t projDim;      // n * n + 2 * n

    // 多核切分
    uint32_t usedCoreNum;
    uint32_t rowsPerCore;

    // 可选 gamma 是否存在
    uint32_t hasGamma;

    // 两个属性
    float normEps;
    float hcEps;

    float invND;
};