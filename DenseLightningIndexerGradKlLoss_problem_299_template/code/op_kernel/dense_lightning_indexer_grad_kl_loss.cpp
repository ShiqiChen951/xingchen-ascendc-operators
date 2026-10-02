// Kernel侧核函数实现
#include "kernel_operator.h"

#include "dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_dense_lightning_indexer_grad_kl_loss.h"

using namespace AscendC;

__aicore__ inline float ToFloatScalar(float v) {
    return v;
}

__aicore__ inline float ToFloatScalar(half v) {
    return static_cast<float>(v);
}

template <class DT_QUERY>
class KernelDenseLightningIndexerGradKlLoss {
public:
    __aicore__ inline KernelDenseLightningIndexerGradKlLoss() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index, GM_ADDR weights, GM_ADDR d_query_index, GM_ADDR d_key_index, GM_ADDR d_weights, GM_ADDR loss, uint32_t length, const DenseLightningIndexerGradKlLossTilingData &tiling) {
        length_ = length;

        batchSize_ = tiling.batchSize;
        querySeqLen_ = tiling.querySeqLen;
        keySeqLen_ = tiling.keySeqLen;

        queryHeadNum_ = tiling.queryHeadNum;
        indexHeadNum_ = tiling.indexHeadNum;

        headDim_ = tiling.headDim;
        scaleValue_ = tiling.scaleValue;
        queryGm_.SetGlobalBuffer(
            (__gm__ DT_QUERY *)query,
            batchSize_ * querySeqLen_ * queryHeadNum_ * headDim_
        );

        keyGm_.SetGlobalBuffer(
            (__gm__ DT_QUERY *)key,
            batchSize_ * keySeqLen_ * queryHeadNum_ * headDim_
        );

        queryIndexGm_.SetGlobalBuffer(
            (__gm__ DT_QUERY *)query_index,
            batchSize_ * querySeqLen_ * indexHeadNum_ * headDim_
        );

        keyIndexGm_.SetGlobalBuffer(
            (__gm__ DT_QUERY *)key_index,
            batchSize_ * keySeqLen_ * headDim_
        );

        weightsGm_.SetGlobalBuffer(
            (__gm__ DT_QUERY *)weights,
            batchSize_ * querySeqLen_ * indexHeadNum_
        );

        dQueryIndexGm_.SetGlobalBuffer(
            (__gm__ DT_QUERY *)d_query_index,
            batchSize_ * querySeqLen_ * indexHeadNum_ * headDim_
        );

        dKeyIndexGm_.SetGlobalBuffer(
            (__gm__ DT_QUERY *)d_key_index,
            batchSize_ * keySeqLen_ * headDim_
        );

        dWeightsGm_.SetGlobalBuffer(
            (__gm__ DT_QUERY *)d_weights,
            batchSize_ * querySeqLen_ * indexHeadNum_
        );

        lossGm_.SetGlobalBuffer(
            (__gm__ float *)loss,
            1
        );

        pipe_.InitBuffer(calcBuf_, 256);
        pipe_.InitBuffer(
            gradBuf_,
            indexHeadNum_ * (headDim_ + 1) * sizeof(float)
        );
        pipe_.InitBuffer(
            mainSoftmaxBuf_,
            queryHeadNum_ * 2 * sizeof(float)
        );
    }
    __aicore__ inline void Process()
{
    if (GetBlockIdx() != 0) {
        return;
    }

    float totalLoss = 0.0f;

    for (uint32_t b = 0;
         b < batchSize_;
         ++b)
    {
        // ==========================================
        // 0. 清零当前 batch 的 dKeyIndex
        // ==========================================

        for (uint32_t j = 0;
             j < keySeqLen_;
             ++j)
        {
            for (uint32_t d = 0;
                 d < headDim_;
                 ++d)
            {
                dKeyIndexGm_.SetValue(
                    KeyIndexOffset(
                        b,
                        j,
                        d
                    ),
                    FromFloatScalar(
                        0.0f
                    )
                );
            }
        }


        // ==========================================
        // dWeights + dQueryIndex + dKeyIndex + loss
        // ==========================================

        for (uint32_t s = 0;
             s < querySeqLen_;
             ++s)
        {
            // ==========================================
            // Indexer Softmax：
            // 当前 (b,s) 的 max / sum
            // ==========================================

            float maxI = 0.0f;
            float sumI = 0.0f;

            ComputeIndexerMaxAndSum(
                b,
                s,
                maxI,
                sumI
            );

            // ==========================================
            // Main Attention Softmax：
            //
            // 对当前 (b,s)，提前缓存：
            //
            // mainMax[h]
            // mainSum[h]
            //
            // 后面所有 j 都复用
            // ==========================================

            LocalTensor<float> mainSoftmaxLocal =
                mainSoftmaxBuf_.Get<float>();

            uint32_t mainMaxBase = 0;

            uint32_t mainSumBase =
                queryHeadNum_;


            for (uint32_t h = 0;
                 h < queryHeadNum_;
                 ++h)
            {
                // ------------------------------
                // 当前 head 的最大值
                // ------------------------------

                float mainMax =
                    ComputeMainMax(
                        b,
                        s,
                        h
                    );


                // ------------------------------
                // 当前 head 的 exp sum
                // ------------------------------

                float mainSum =
                    ComputeMainExpSum(
                        b,
                        s,
                        h,
                        mainMax
                    );


                // ------------------------------
                // 保存到 UB
                // ------------------------------

                mainSoftmaxLocal.SetValue(
                    mainMaxBase + h,
                    mainMax
                );

                mainSoftmaxLocal.SetValue(
                    mainSumBase + h,
                    mainSum
                );
            }


            // ==========================================
            // gradLocal 布局
            //
            // [0, indexHeadNum_)
            //      -> dWeight[h]
            //
            // [indexHeadNum_, ...)
            //      -> dQueryIndex[h][d]
            // ==========================================

            LocalTensor<float> gradLocal =
                gradBuf_.Get<float>();

            uint32_t dWeightBase = 0;

            uint32_t dQueryBase =
                indexHeadNum_;


            // ==========================================
            // 1. 清零当前 query 的
            //    dWeight / dQueryIndex
            // ==========================================

            for (uint32_t h = 0;
                 h < indexHeadNum_;
                 ++h)
            {
                gradLocal.SetValue(
                    dWeightBase + h,
                    0.0f
                );

                for (uint32_t d = 0;
                     d < headDim_;
                     ++d)
                {
                    uint32_t offset =
                        dQueryBase
                        + h * headDim_
                        + d;

                    gradLocal.SetValue(
                        offset,
                        0.0f
                    );
                }
            }


            // ==========================================
            // 2. 遍历所有 key
            //
            // 每一个 j：
            //
            // prob / p / dI 只计算一次
            //
            // 同时更新：
            // loss
            // dWeight
            // dQueryIndex
            // dKeyIndex
            // ==========================================

            for (uint32_t j = 0;
                 j < keySeqLen_;
                 ++j)
            {
                // --------------------------------------
                // Indexer预测概率
                // --------------------------------------

                float prob =
                    ComputeIndexerSoftmax(
                        b,
                        s,
                        j,
                        maxI,
                        sumI
                    );


                // --------------------------------------
                // Main Attention target概率
                //
                // 这里改成Cached版本
                //
                // 不再重复计算MainMax/MainExpSum
                // --------------------------------------

                float p =
                    ComputeTargetProbabilityCached(
                        b,
                        s,
                        j,
                        mainSoftmaxLocal
                    );


                // --------------------------------------
                // loss
                // --------------------------------------

                constexpr float EPS =
                    1.0e-12f;

                float ratio =
                    (p + EPS)
                    / (prob + EPS);

                totalLoss +=
                    p * LogScalar(
                        ratio
                    );


                // --------------------------------------
                // dI
                // --------------------------------------

                float dI =
                    prob - p;


                // ======================================
                // 当前 dI 同时作用于所有 index head
                // ======================================

                for (uint32_t h = 0;
                     h < indexHeadNum_;
                     ++h)
                {
                    // ----------------------------------
                    // S[h,j]
                    // ----------------------------------

                    float dot =
                        ComputeIndexDot(
                            b,
                            s,
                            h,
                            j
                        );


                    // ----------------------------------
                    // ReLU反向
                    // ----------------------------------

                    if (dot <= 0.0f) {
                        continue;
                    }


                    // ==================================
                    // dWeight[h]
                    //
                    // += dI[j] * ReLU(S[h,j])
                    // ==================================

                    uint32_t weightGradOffset =
                        dWeightBase + h;

                    float oldDWeight =
                        gradLocal.GetValue(
                            weightGradOffset
                        );

                    oldDWeight +=
                        dI * dot;

                    gradLocal.SetValue(
                        weightGradOffset,
                        oldDWeight
                    );


                    // ==================================
                    // dS[h,j]
                    //
                    // = weight[h] * dI[j]
                    // ==================================

                    DT_QUERY weightValue =
                        weightsGm_.GetValue(
                            WeightOffset(
                                b,
                                s,
                                h
                            )
                        );

                    float weight =
                        ToFloatScalar(
                            weightValue
                        );

                    float dS =
                        weight * dI;


                    // ==================================
                    // 同时更新
                    //
                    // dQueryIndex[h,d]
                    // dKeyIndex[j,d]
                    // ==================================

                    for (uint32_t d = 0;
                         d < headDim_;
                         ++d)
                    {
                        // ==============================
                        // dQueryIndex
                        //
                        // += dS * keyIndex
                        // ==============================

                        DT_QUERY kValue =
                            keyIndexGm_.GetValue(
                                KeyIndexOffset(
                                    b,
                                    j,
                                    d
                                )
                            );

                        float k =
                            ToFloatScalar(
                                kValue
                            );


                        uint32_t queryGradOffset =
                            dQueryBase
                            + h * headDim_
                            + d;


                        float oldQueryGrad =
                            gradLocal.GetValue(
                                queryGradOffset
                            );


                        oldQueryGrad +=
                            dS * k;


                        gradLocal.SetValue(
                            queryGradOffset,
                            oldQueryGrad
                        );


                        // ==============================
                        // dKeyIndex
                        //
                        // += dS * queryIndex
                        // ==============================

                        DT_QUERY qValue =
                            queryIndexGm_.GetValue(
                                QueryIndexOffset(
                                    b,
                                    s,
                                    h,
                                    d
                                )
                            );

                        float q =
                            ToFloatScalar(
                                qValue
                            );


                        uint32_t keyGradOffset =
                            KeyIndexOffset(
                                b,
                                j,
                                d
                            );


                        DT_QUERY oldKeyGradValue =
                            dKeyIndexGm_.GetValue(
                                keyGradOffset
                            );


                        float oldKeyGrad =
                            ToFloatScalar(
                                oldKeyGradValue
                            );


                        oldKeyGrad +=
                            dS * q;


                        dKeyIndexGm_.SetValue(
                            keyGradOffset,
                            FromFloatScalar(
                                oldKeyGrad
                            )
                        );
                    }
                }
            }


            // ==========================================
            // 3. 当前 query 的
            //    dWeights / dQueryIndex 写回GM
            // ==========================================

            for (uint32_t h = 0;
                 h < indexHeadNum_;
                 ++h)
            {
                // --------------------------------------
                // dWeights
                // --------------------------------------

                float dWeight =
                    gradLocal.GetValue(
                        dWeightBase + h
                    );

                dWeightsGm_.SetValue(
                    WeightOffset(
                        b,
                        s,
                        h
                    ),
                    FromFloatScalar(
                        dWeight
                    )
                );


                // --------------------------------------
                // dQueryIndex
                // --------------------------------------

                for (uint32_t d = 0;
                     d < headDim_;
                     ++d)
                {
                    uint32_t queryGradOffset =
                        dQueryBase
                        + h * headDim_
                        + d;

                    float grad =
                        gradLocal.GetValue(
                            queryGradOffset
                        );

                    dQueryIndexGm_.SetValue(
                        QueryIndexOffset(
                            b,
                            s,
                            h,
                            d
                        ),
                        FromFloatScalar(
                            grad
                        )
                    );
                }
            }
        }
    }


    // ==========================================
    // 最终 KL loss 写回
    // ==========================================

    lossGm_.SetValue(
        0,
        totalLoss
    );
}

private:
    uint32_t length_;

    uint32_t batchSize_;
    uint32_t querySeqLen_;
    uint32_t keySeqLen_;

    uint32_t queryHeadNum_;
    uint32_t indexHeadNum_;

    uint32_t headDim_;

    float scaleValue_;

    GlobalTensor<DT_QUERY> queryGm_;
    GlobalTensor<DT_QUERY> keyGm_;

    GlobalTensor<DT_QUERY> queryIndexGm_;
    GlobalTensor<DT_QUERY> keyIndexGm_;

    GlobalTensor<DT_QUERY> weightsGm_;

    GlobalTensor<DT_QUERY> dQueryIndexGm_;
    GlobalTensor<DT_QUERY> dKeyIndexGm_;
    GlobalTensor<DT_QUERY> dWeightsGm_;

    GlobalTensor<float> lossGm_;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    TBuf<QuePosition::VECCALC> gradBuf_;
    TBuf<QuePosition::VECCALC> mainSoftmaxBuf_;

    __aicore__ inline uint32_t QueryOffset(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        uint32_t d)
    {
        return (((b * querySeqLen_ + s)
                * queryHeadNum_ + h)
                * headDim_ + d);
    }
    __aicore__ inline uint32_t KeyOffset(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        uint32_t d)
    {
        return (((b * keySeqLen_ + s)
                * queryHeadNum_ + h)
                * headDim_ + d);
    }
    __aicore__ inline uint32_t QueryIndexOffset(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        uint32_t d)
    {
        return (((b * querySeqLen_ + s)
                * indexHeadNum_ + h)
                * headDim_ + d);
    }
    __aicore__ inline uint32_t KeyIndexOffset(
        uint32_t b,
        uint32_t s,
        uint32_t d)
    {
        return ((b * keySeqLen_ + s)
                * headDim_ + d);
    }
    __aicore__ inline uint32_t WeightOffset(
        uint32_t b,
        uint32_t s,
        uint32_t h)
    {
        return ((b * querySeqLen_ + s)
                * indexHeadNum_ + h);
    }
    __aicore__ inline float ComputeIndexerScore(
        uint32_t b,
        uint32_t s,
        uint32_t j)
    {
        float indexerScore = 0.0f;

        for (uint32_t h = 0;
            h < indexHeadNum_;
            ++h)
        {
            float dot =
                ComputeIndexDot(
                    b,
                    s,
                    h,
                    j
                );

            float reluValue =
                dot > 0.0f
                    ? dot
                    : 0.0f;

            DT_QUERY weightValue =
                weightsGm_.GetValue(
                    WeightOffset(
                        b,
                        s,
                        h
                    )
                );

            float weight =
                ToFloatScalar(weightValue);

            indexerScore +=
                weight * reluValue;
        }

        return indexerScore;
    }
    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> temp =
            calcBuf_.Get<float>();

        temp.SetValue(0, x);

        AscendC::Exp(
            temp,
            temp,
            1
        );

        return temp.GetValue(0);
    }

    __aicore__ inline float ComputeIndexerMax(
        uint32_t b,
        uint32_t s)
    {
        float maxValue = -3.4e38f;

        for (uint32_t j = 0;
            j < keySeqLen_;
            ++j)
        {
            float score =
                ComputeIndexerScore(
                    b,
                    s,
                    j
                );

            if (score > maxValue) {
                maxValue = score;
            }
        }

        return maxValue;
    }
    __aicore__ inline float ComputeIndexerExpSum(
        uint32_t b,
        uint32_t s,
        float maxValue)
    {
        float sum = 0.0f;

        for (uint32_t j = 0;
            j < keySeqLen_;
            ++j)
        {
            float score =
                ComputeIndexerScore(
                    b,
                    s,
                    j
                );

            float e =
                ExpScalar(
                    score - maxValue
                );

            sum += e;
        }

        return sum;
    }
    __aicore__ inline float ComputeIndexerSoftmax(
        uint32_t b,
        uint32_t s,
        uint32_t j,
        float maxValue,
        float expSum)
    {
        float score =
            ComputeIndexerScore(
                b,
                s,
                j
            );

        float numerator =
            ExpScalar(
                score - maxValue
            );

        return numerator / expSum;
    }
    __aicore__ inline float ComputeMainScore(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        uint32_t j)
    {
        float dot = 0.0f;

        // query[b, s, h, :] · key[b, j, h, :]
        for (uint32_t d = 0; d < headDim_; ++d) {

            DT_QUERY qValue =
                queryGm_.GetValue(
                    QueryOffset(b, s, h, d));

            DT_QUERY kValue =
                keyGm_.GetValue(
                    KeyOffset(b, j, h, d));

            float q = ToFloatScalar(qValue);
            float k = ToFloatScalar(kValue);

            dot += q * k;
        }

        // Attention scale
        return dot * scaleValue_;
    }
    __aicore__ inline float ComputeMainMax(
        uint32_t b,
        uint32_t s,
        uint32_t h)
    {
        float maxValue = -3.4e38f;

        for (uint32_t j = 0;
            j < keySeqLen_;
            ++j)
        {
            float score =
                ComputeMainScore(
                    b,
                    s,
                    h,
                    j
                );

            if (score > maxValue) {
                maxValue = score;
            }
        }

        return maxValue;
    }
    __aicore__ inline float ComputeMainExpSum(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        float maxValue)
    {
        float sum = 0.0f;

        for (uint32_t j = 0;
            j < keySeqLen_;
            ++j)
        {
            float score =
                ComputeMainScore(
                    b,
                    s,
                    h,
                    j
                );

            float e =
                ExpScalar(
                    score - maxValue
                );

            sum += e;
        }

        return sum;
    }
    __aicore__ inline float ComputeMainSoftmax(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        uint32_t j,
        float maxValue,
        float expSum)
    {
        float score =
            ComputeMainScore(
                b,
                s,
                h,
                j
            );

        float numerator =
            ExpScalar(
                score - maxValue
            );

        return numerator / expSum;
    }
    __aicore__ inline float ComputeTargetProbability(
        uint32_t b,
        uint32_t s,
        uint32_t j)
    {
        float sumProbability = 0.0f;

        for (uint32_t h = 0;
            h < queryHeadNum_;
            ++h)
        {
            float maxValue =
                ComputeMainMax(
                    b,
                    s,
                    h
                );

            float expSum =
                ComputeMainExpSum(
                    b,
                    s,
                    h,
                    maxValue
                );

            float probability =
                ComputeMainSoftmax(
                    b,
                    s,
                    h,
                    j,
                    maxValue,
                    expSum
                );

            sumProbability += probability;
        }

        // 所有主注意力 head 求和后做 L1 normalize
        float invHeadNum = 0.0f;

        if (queryHeadNum_ == 32) {
            invHeadNum = 1.0f / 32.0f;
        } else if (queryHeadNum_ == 64) {
            invHeadNum = 1.0f / 64.0f;
        } else {
            invHeadNum = 1.0f / 128.0f;
        }

        return sumProbability * invHeadNum;
    }
    __aicore__ inline DT_QUERY FromFloatScalar(float v)
    {
        return static_cast<DT_QUERY>(v);
    }
    __aicore__ inline float ComputeDI(
        uint32_t b,
        uint32_t s,
        uint32_t j,
        float maxI,
        float sumI)
    {
        // Indexer预测分布
        float prob =
            ComputeIndexerSoftmax(
                b,
                s,
                j,
                maxI,
                sumI
            );

        // 主注意力target分布
        float p =
            ComputeTargetProbability(
                b,
                s,
                j
            );

        // dI = softmax(I) - p
        return prob - p;
    }
    __aicore__ inline float ComputeWeightGrad(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        float maxI,
        float sumI)
    {
        float grad = 0.0f;

        for (uint32_t j = 0;
            j < keySeqLen_;
            ++j)
        {
            float dot =
                ComputeIndexDot(
                    b,
                    s,
                    h,
                    j
                );

            float reluValue =
                dot > 0.0f
                    ? dot
                    : 0.0f;

            float dI =
                ComputeDI(
                    b,
                    s,
                    j,
                    maxI,
                    sumI
                );

            grad +=
                dI * reluValue;
        }

        return grad;
    }
    __aicore__ inline float ComputeIndexDot(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        uint32_t j)
    {
        float dot = 0.0f;

        for (uint32_t d = 0;
            d < headDim_;
            ++d)
        {
            DT_QUERY qValue =
                queryIndexGm_.GetValue(
                    QueryIndexOffset(
                        b,
                        s,
                        h,
                        d
                    )
                );

            DT_QUERY kValue =
                keyIndexGm_.GetValue(
                    KeyIndexOffset(
                        b,
                        j,
                        d
                    )
                );

            float q =
                ToFloatScalar(qValue);

            float k =
                ToFloatScalar(kValue);

            dot += q * k;
        }

        return dot;
    }
    __aicore__ inline float ComputeDS(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        uint32_t j,
        float maxI,
        float sumI)
    {
        // S[h,j]
        float dot =
            ComputeIndexDot(
                b,
                s,
                h,
                j
            );

        // ReLU导数
        // S <= 0 时，梯度为0
        if (dot <= 0.0f) {
            return 0.0f;
        }

        // dI[j]
        float dI =
            ComputeDI(
                b,
                s,
                j,
                maxI,
                sumI
            );

        // weight[h]
        DT_QUERY weightValue =
            weightsGm_.GetValue(
                WeightOffset(
                    b,
                    s,
                    h
                )
            );

        float weight =
            ToFloatScalar(weightValue);

        // dS[h,j] = weight[h] * dI[j]
        return weight * dI;
    }
    __aicore__ inline float ComputeQueryIndexGrad(
        uint32_t b,
        uint32_t s,
        uint32_t h,
        uint32_t d,
        float maxI,
        float sumI)
    {
        float grad = 0.0f;

        for (uint32_t j = 0;
            j < keySeqLen_;
            ++j)
        {
            // dS[h,j]
            float dS =
                ComputeDS(
                    b,
                    s,
                    h,
                    j,
                    maxI,
                    sumI
                );

            // keyIndex[b,j,0,d]
            DT_QUERY kValue =
                keyIndexGm_.GetValue(
                    KeyIndexOffset(
                        b,
                        j,
                        d
                    )
                );

            float k =
                ToFloatScalar(kValue);

            // dQi[h,d] += dS[h,j] * Ki[j,d]
            grad += dS * k;
        }

        return grad;
    }
    __aicore__ inline float ComputeKeyIndexGrad(
        uint32_t b,
        uint32_t j,
        uint32_t d)
    {
        float grad = 0.0f;

        // keyIndex[b,j,0,d] 被所有 query token 使用
        for (uint32_t s = 0;
            s < querySeqLen_;
            ++s)
        {
            // 每个 query token 都有自己的一套 Indexer softmax
            float maxI = 0.0f;
            float sumI = 0.0f;

            ComputeIndexerMaxAndSum(
                b,
                s,
                maxI,
                sumI
            );

            // 遍历所有 index head
            for (uint32_t h = 0;
                h < indexHeadNum_;
                ++h)
            {
                // dS[b,s,h,j]
                float dS =
                    ComputeDS(
                        b,
                        s,
                        h,
                        j,
                        maxI,
                        sumI
                    );

                // queryIndex[b,s,h,d]
                DT_QUERY qValue =
                    queryIndexGm_.GetValue(
                        QueryIndexOffset(
                            b,
                            s,
                            h,
                            d
                        )
                    );

                float q =
                    ToFloatScalar(
                        qValue
                    );

                // dK[j,d] += dS[s,h,j] * Q[s,h,d]
                grad += dS * q;
            }
        }

        return grad;
    }
    __aicore__ inline float LogScalar(float x)
    {
        LocalTensor<float> temp =
            calcBuf_.Get<float>();

        temp.SetValue(0, x);

        AscendC::Log(
            temp,
            temp,
            1
        );

        return temp.GetValue(0);
    }
    __aicore__ inline float ComputeQueryLoss(
        uint32_t b,
        uint32_t s,
        float maxI,
        float sumI)
    {
        float loss = 0.0f;

        constexpr float EPS = 1.0e-12f;

        for (uint32_t j = 0;
            j < keySeqLen_;
            ++j)
        {
            // Indexer预测概率 softmax(I)[j]
            float prob =
                ComputeIndexerSoftmax(
                    b,
                    s,
                    j,
                    maxI,
                    sumI
                );

            // 主注意力生成的target概率 p[j]
            float p =
                ComputeTargetProbability(
                    b,
                    s,
                    j
                );

            // KL:
            //
            // p * log((p + eps) / (prob + eps))

            float ratio =
                (p + EPS) /
                (prob + EPS);

            loss +=
                p * LogScalar(ratio);
        }

        return loss;
    }
    __aicore__ inline float ComputeTargetProbabilityCached(
        uint32_t b,
        uint32_t s,
        uint32_t j,
        LocalTensor<float> mainSoftmaxLocal)
    {
        float sumProbability = 0.0f;

        uint32_t maxBase = 0;
        uint32_t sumBase = queryHeadNum_;

        for (uint32_t h = 0;
            h < queryHeadNum_;
            ++h)
        {
            float maxValue =
                mainSoftmaxLocal.GetValue(
                    maxBase + h
                );

            float expSum =
                mainSoftmaxLocal.GetValue(
                    sumBase + h
                );

            float probability =
                ComputeMainSoftmax(
                    b,
                    s,
                    h,
                    j,
                    maxValue,
                    expSum
                );

            sumProbability += probability;
        }


        float invHeadNum = 0.0f;

        if (queryHeadNum_ == 32) {
            invHeadNum = 1.0f / 32.0f;
        } else if (queryHeadNum_ == 64) {
            invHeadNum = 1.0f / 64.0f;
        } else {
            invHeadNum = 1.0f / 128.0f;
        }

        return sumProbability * invHeadNum;
    }
    __aicore__ inline void ComputeIndexerMaxAndSum(
        uint32_t b,
        uint32_t s,
        float &maxValue,
        float &expSum)
    {
        maxValue = -3.4e38f;
        expSum = 0.0f;

        for (uint32_t j = 0;
            j < keySeqLen_;
            ++j)
        {
            float score =
                ComputeIndexerScore(
                    b,
                    s,
                    j
                );

            if (score > maxValue) {
                if (maxValue < -3.0e38f) {
                    expSum = 1.0f;
                } else {
                    expSum =
                        expSum *
                        ExpScalar(
                            maxValue - score
                        )
                        + 1.0f;
                }

                maxValue = score;
            }
            else {
                expSum +=
                    ExpScalar(
                        score - maxValue
                    );
            }
        }
    }
};

template <typename DT_QUERY>
 __global__ __aicore__ void dense_lightning_indexer_grad_kl_loss(GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index, GM_ADDR weights, GM_ADDR d_query_index, GM_ADDR d_key_index, GM_ADDR d_weights, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DenseLightningIndexerGradKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(DenseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
    KernelDenseLightningIndexerGradKlLoss<DT_QUERY> op;
    op.Init(query, key, query_index, key_index, weights, d_query_index, d_key_index, d_weights, loss, tiling_data.length, tiling_data);
    op.Process();
}
