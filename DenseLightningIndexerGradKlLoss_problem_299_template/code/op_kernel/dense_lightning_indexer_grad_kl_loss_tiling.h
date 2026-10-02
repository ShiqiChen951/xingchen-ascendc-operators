// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct DenseLightningIndexerGradKlLossTilingData {
    uint32_t length;
    uint32_t batchSize;     // B
    uint32_t querySeqLen;   // S1
    uint32_t keySeqLen;     // S2

    uint32_t queryHeadNum;  // N1
    uint32_t indexHeadNum;  // Nidx1

    uint32_t headDim;       // D

    float scaleValue;       // 主注意力缩放系数
};