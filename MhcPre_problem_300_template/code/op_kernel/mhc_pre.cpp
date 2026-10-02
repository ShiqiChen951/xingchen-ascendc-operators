// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_pre_tiling.h"
#include "tiling_key_mhc_pre.h"

using namespace AscendC;

constexpr uint32_t MHC_PRE_TILE_ELEMS = 4096;

__aicore__ inline float ToFloatScalar(float v)
{
    return v;
}

__aicore__ inline float ToFloatScalar(half v)
{
    return static_cast<float>(v);
}

template <class DT_X>
class KernelMhcPre {
public:
    __aicore__ inline KernelMhcPre() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes, uint32_t length, const MhcPreTilingData &tiling) {
        length_ = length;
        // -------------------------
        // 新增 tiling 参数
        // -------------------------
        batchSize_ = tiling.batchSize;
        seqLen_ = tiling.seqLen;
        n_ = tiling.n;
        hiddenDim_ = tiling.hiddenDim;

        rowNum_ = tiling.rowNum;
        nD_ = tiling.nD;
        projDim_ = tiling.projDim;

        usedCoreNum_ = tiling.usedCoreNum;
        rowsPerCore_ = tiling.rowsPerCore;

        hasGamma_ = tiling.hasGamma;

        normEps_ = tiling.normEps;
        hcEps_ = tiling.hcEps;
        invND_ = tiling.invND;

        // -------------------------
        // 当前核处理范围
        // -------------------------
        coreId_ = GetBlockIdx();

        startRow_ = coreId_ * rowsPerCore_;

        endRow_ = startRow_ + rowsPerCore_;

        if (endRow_ > rowNum_) {
            endRow_ = rowNum_;
        }

        // -------------------------
        // 输入 GlobalTensor
        // -------------------------
        xGm_.SetGlobalBuffer(
            (__gm__ DT_X *)x,
            length_);

        phiGm_.SetGlobalBuffer(
            (__gm__ float *)phi,
            projDim_ * nD_);

        alphaGm_.SetGlobalBuffer(
            (__gm__ float *)alpha,
            3);

        biasGm_.SetGlobalBuffer(
            (__gm__ float *)bias,
            projDim_);

        if (hasGamma_ != 0) {
            gammaGm_.SetGlobalBuffer(
                (__gm__ float *)gamma,
                nD_);
        }

        // -------------------------
        // 输出 GlobalTensor
        // -------------------------
        hInGm_.SetGlobalBuffer(
            (__gm__ DT_X *)hIn,
            rowNum_ * hiddenDim_);

        hPostGm_.SetGlobalBuffer(
            (__gm__ float *)hPost,
            rowNum_ * n_);

        hResGm_.SetGlobalBuffer(
            (__gm__ float *)hRes,
            rowNum_ * n_ * n_);

        pipe_.InitBuffer(
            xBuf_,
            MHC_PRE_TILE_ELEMS * sizeof(DT_X));

        pipe_.InitBuffer(
            wBuf_,
            80 * sizeof(float));

        pipe_.InitBuffer(
            hPreBuf_,
            8 * sizeof(float));

        pipe_.InitBuffer(
            calcBuf_,
            16 * sizeof(float));

        pipe_.InitBuffer(
            hInBuf_,
            MHC_PRE_TILE_ELEMS * sizeof(float));

        pipe_.InitBuffer(
            xRowBuf_,
            MHC_PRE_TILE_ELEMS * sizeof(DT_X));
        
        pipe_.InitBuffer(sigmoidSrcBuf_, 32);
        pipe_.InitBuffer(sigmoidDstBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = startRow_;
            row < endRow_;
            ++row) {

            // 1. RMSNorm
            float invRms =
                ComputeInvRms(row);

            // 2. hMix * invRms = w
            ComputeWAll(
                row,
                invRms);

            // 3. hPre / hPost
            ComputeHPreAndHPost(
                row);

            // 4. hRes
            ComputeHRes(
                row);

            // 5. hIn
            ComputeHIn(
                row);
        }
    }

    __aicore__ inline float ComputeInvRms(uint32_t row)
    {
        LocalTensor<DT_X> xLocal =
            xBuf_.Get<DT_X>();

        // x[row] 在 GM 中的起始位置
        uint32_t xOffset = row * nD_;

        float squareSum = 0.0f;

        // 分块遍历 nD
        for (uint32_t offset = 0;
            offset < nD_;
            offset += MHC_PRE_TILE_ELEMS) {

            uint32_t curCount =
                nD_ - offset;

            if (curCount > MHC_PRE_TILE_ELEMS) {
                curCount = MHC_PRE_TILE_ELEMS;
            }

            // GM -> UB
            DataCopy(
                xLocal,
                xGm_[xOffset + offset],
                curCount);

            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);

            // 当前 tile 的平方和
            for (uint32_t i = 0;
                i < curCount;
                ++i) {

                float value =
                    ToFloatScalar(
                        xLocal.GetValue(i));

                squareSum +=
                    value * value;
            }
        }

        float meanSquare =
            squareSum * invND_;

        float invRms =
            1.0f /
            sqrt(meanSquare + normEps_);

        return invRms;
    }

    __aicore__ inline void ComputeWAll(
    uint32_t row,
    float invRms)
{
    LocalTensor<float> wLocal =
        wBuf_.Get<float>();

    uint32_t xOffset =
        row * nD_;

    for (uint32_t j = 0;
         j < projDim_;
         ++j) {

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        uint32_t k = 0;

        for (; k + 3 < nD_; k += 4) {

            float x0 =
                ToFloatScalar(
                    xGm_.GetValue(
                        xOffset + k));

            float x1 =
                ToFloatScalar(
                    xGm_.GetValue(
                        xOffset + k + 1));

            float x2 =
                ToFloatScalar(
                    xGm_.GetValue(
                        xOffset + k + 2));

            float x3 =
                ToFloatScalar(
                    xGm_.GetValue(
                        xOffset + k + 3));

            if (hasGamma_ != 0) {
                x0 *= gammaGm_.GetValue(k);
                x1 *= gammaGm_.GetValue(k + 1);
                x2 *= gammaGm_.GetValue(k + 2);
                x3 *= gammaGm_.GetValue(k + 3);
            }

            uint32_t phiOffset =
                j * nD_ + k;

            sum0 +=
                x0 *
                phiGm_.GetValue(
                    phiOffset);

            sum1 +=
                x1 *
                phiGm_.GetValue(
                    phiOffset + 1);

            sum2 +=
                x2 *
                phiGm_.GetValue(
                    phiOffset + 2);

            sum3 +=
                x3 *
                phiGm_.GetValue(
                    phiOffset + 3);
        }

        float sum =
            (sum0 + sum1) +
            (sum2 + sum3);

        for (; k < nD_; ++k) {

            float xVal =
                ToFloatScalar(
                    xGm_.GetValue(
                        xOffset + k));

            if (hasGamma_ != 0) {
                xVal *=
                    gammaGm_.GetValue(k);
            }

            sum +=
                xVal *
                phiGm_.GetValue(
                    j * nD_ + k);
        }

        wLocal.SetValue(
            j,
            sum * invRms);
    }
}

__aicore__ inline float SigmoidScalar(float x)
{
    LocalTensor<float> src =
        sigmoidSrcBuf_.Get<float>();

    LocalTensor<float> dst =
        sigmoidDstBuf_.Get<float>();

    // Scalar 写 UB
    src.SetValue(0, -x);

    // Scalar -> Vector
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);

    // exp(-x)
    Exp(
        dst,
        src,
        1);

    // Vector -> Scalar
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);

    float expValue =
        dst.GetValue(0);

    return 1.0f /
        (1.0f + expValue);
}

    __aicore__ inline void ComputeHPreAndHPost(
        uint32_t row)
    {
        LocalTensor<float> wLocal =
            wBuf_.Get<float>();

        LocalTensor<float> hPreLocal =
            hPreBuf_.Get<float>();

        // alpha.shape = [3]
        float alpha0 =
            ToFloatScalar(
                alphaGm_.GetValue(0));

        float alpha1 =
            ToFloatScalar(
                alphaGm_.GetValue(1));

        // -------------------------
        // hPre
        // w[0 : n]
        // -------------------------
        for (uint32_t i = 0;
            i < n_;
            ++i) {

            float pPre =
                wLocal.GetValue(i);

            float bias0 =
                ToFloatScalar(
                    biasGm_.GetValue(i));

            float z =
                pPre * alpha0 +
                bias0;

            float hPre =
                SigmoidScalar(z) +
                hcEps_;

            // 暂存在 UB
            // 后面 hIn 会直接使用
            hPreLocal.SetValue(
                i,
                hPre);
        }

        // -------------------------
        // hPost
        // w[n : 2n]
        // -------------------------
        uint32_t hPostOffset =
            row * n_;

        for (uint32_t i = 0;
            i < n_;
            ++i) {

            float pPost =
                wLocal.GetValue(
                    n_ + i);

            float bias1 =
                ToFloatScalar(
                    biasGm_.GetValue(
                        n_ + i));

            float z =
                pPost * alpha1 +
                bias1;

            float hPost =
                2.0f *
                SigmoidScalar(z);

            hPostGm_.SetValue(
                hPostOffset + i,
                hPost);
        }
    }

    __aicore__ inline void ComputeHRes(uint32_t row)
    {
        LocalTensor<float> wLocal =
            wBuf_.Get<float>();

        float alpha2 =
            ToFloatScalar(
                alphaGm_.GetValue(2));

        uint32_t hResOffset =
            row * n_ * n_;

        uint32_t pResOffset =
            2 * n_;

        for (uint32_t i = 0;
            i < n_ * n_;
            ++i) {

            float pRes =
                wLocal.GetValue(
                    pResOffset + i);

            float bias2 =
                ToFloatScalar(
                    biasGm_.GetValue(
                        pResOffset + i));

            float hRes =
                pRes * alpha2 +
                bias2;

            hResGm_.SetValue(
                hResOffset + i,
                hRes);
        }
    }

    __aicore__ inline void ComputeHIn(uint32_t row)
    {
        LocalTensor<float> hPreLocal =
            hPreBuf_.Get<float>();

        LocalTensor<float> hInLocal =
            hInBuf_.Get<float>();

        LocalTensor<DT_X> xRowLocal =
            xRowBuf_.Get<DT_X>();

        uint32_t xBaseOffset =
            row * nD_;

        uint32_t hInBaseOffset =
            row * hiddenDim_;

        // 沿 D 分块
        for (uint32_t dOffset = 0;
            dOffset < hiddenDim_;
            dOffset += MHC_PRE_TILE_ELEMS) {

            uint32_t curCount =
                hiddenDim_ - dOffset;

            if (curCount > MHC_PRE_TILE_ELEMS) {
                curCount = MHC_PRE_TILE_ELEMS;
            }

            // 先把输出累加区清零
            for (uint32_t d = 0;
                d < curCount;
                ++d) {

                hInLocal.SetValue(
                    d,
                    0.0f);
            }

            // 对 n 个分支加权累加
            for (uint32_t i = 0;
                i < n_;
                ++i) {

                float weight =
                    hPreLocal.GetValue(i);

                // x[b,s,i,dOffset] 的起点
                uint32_t xOffset =
                    xBaseOffset
                    + i * hiddenDim_
                    + dOffset;

                // GM -> UB
                DataCopy(
                    xRowLocal,
                    xGm_[xOffset],
                    curCount);

                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);

                // 累加：
                // hIn[d] += hPre[i] * x[i,d]
                for (uint32_t d = 0;
                    d < curCount;
                    ++d) {

                    float xVal =
                        ToFloatScalar(
                            xRowLocal.GetValue(d));

                    float oldVal =
                        hInLocal.GetValue(d);

                    hInLocal.SetValue(
                        d,
                        oldVal +
                        weight * xVal);
                }
            }

            // 写回 hIn
            for (uint32_t d = 0;
                d < curCount;
                ++d) {

                float value =
                    hInLocal.GetValue(d);

                hInGm_.SetValue(
                    hInBaseOffset
                    + dOffset
                    + d,
                    static_cast<DT_X>(value));
            }
        }
    }

private:
    // -------------------------
    // GM Tensor
    // -------------------------
    GlobalTensor<DT_X> xGm_;

    GlobalTensor<float> phiGm_;
    GlobalTensor<float> alphaGm_;
    GlobalTensor<float> biasGm_;
    GlobalTensor<float> gammaGm_;

    GlobalTensor<DT_X> hInGm_;
    GlobalTensor<float> hPostGm_;
    GlobalTensor<float> hResGm_;

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> wBuf_;
    TBuf<QuePosition::VECCALC> hPreBuf_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    TBuf<QuePosition::VECCALC> hInBuf_;
    TBuf<QuePosition::VECCALC> xRowBuf_;

    TBuf<QuePosition::VECCALC> sigmoidSrcBuf_;
    TBuf<QuePosition::VECCALC> sigmoidDstBuf_;
    // -------------------------
    // 模板原有参数
    // -------------------------
    uint32_t length_;

    // -------------------------
    // shape / tiling 参数
    // -------------------------
    uint32_t batchSize_;
    uint32_t seqLen_;
    uint32_t n_;
    uint32_t hiddenDim_;

    uint32_t rowNum_;
    uint32_t nD_;
    uint32_t projDim_;

    uint32_t usedCoreNum_;
    uint32_t rowsPerCore_;
    uint32_t hasGamma_;

    // -------------------------
    // 当前核参数
    // -------------------------
    uint32_t coreId_;
    uint32_t startRow_;
    uint32_t endRow_;

    // -------------------------
    // attr
    // -------------------------
    float normEps_;
    float hcEps_;
    float invND_;
};

template <typename DT_X>
 __global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma, GM_ADDR hIn, GM_ADDR hPost, GM_ADDR hRes, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcPreTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPreTilingData, tiling_data, tiling);
    KernelMhcPre<DT_X> op;
    op.Init(x, phi, alpha, bias, gamma, hIn, hPost, hRes, tiling_data.length, tiling_data);
    op.Process();
}
