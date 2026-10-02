// Kernel-side SparseSoftmax implementation.
//
// Optimized version:
// 1. Support index / ptr grouping.
// 2. Support arbitrary dim.
// 3. Split independent (outer, inner) slices across AI Cores.
// 4. For groupSize <= 256, Exp is calculated only ONCE.
// 5. For larger groups, keep tiled fallback for correctness.
// 6. index / ptr are int32_t, matching the original operator template.

#include "kernel_operator.h"

#include "sparse_softmax_tiling.h"
#include "tiling_key_sparse_softmax.h"

using namespace AscendC;


// ============================================================
// Constants
// ============================================================

constexpr uint32_t SOFTMAX_TILE_ELEMS = 256;

// calcBuf_:
// [0, 255]   : Exp input
// [256, 511] : Exp output
constexpr uint32_t SOFTMAX_CALC_ELEMS =
    SOFTMAX_TILE_ELEMS * 2;


// ============================================================
// Scalar conversion helpers
// ============================================================

__aicore__ inline float ToFloatScalar(float v)
{
    return v;
}

__aicore__ inline float ToFloatScalar(half v)
{
    return static_cast<float>(v);
}


template <typename T>
__aicore__ inline T FromFloatScalar(float v)
{
    return static_cast<T>(v);
}


// ============================================================
// Kernel class
// ============================================================

template <class DT_SRC>
class KernelSparseSoftmax {
public:

    __aicore__ inline KernelSparseSoftmax()
    {
    }


    // ========================================================
    // Init
    // ========================================================

    __aicore__ inline void Init(
        GM_ADDR src,
        GM_ADDR index,
        GM_ADDR ptr,
        GM_ADDR out,
        const SparseSoftmaxTilingData &tiling)
    {
        // ----------------------------------------------------
        // Global tensors
        // ----------------------------------------------------

        srcGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_SRC *>(src),
            tiling.length);

        outGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_SRC *>(out),
            tiling.length);


        // ----------------------------------------------------
        // Tiling information
        // ----------------------------------------------------

        length_ = tiling.length;

        mode_ = tiling.mode;

        dim_ = tiling.dim;

        outerSize_ = tiling.outerSize;

        dimSize_ = tiling.dimSize;

        innerSize_ = tiling.innerSize;

        ptrLength_ = tiling.ptrLength;

        indexLength_ = tiling.indexLength;

        eps_ = tiling.eps;


        // ----------------------------------------------------
        // Optional inputs
        // ----------------------------------------------------

        if (mode_ == 0 &&
            index != nullptr &&
            indexLength_ > 0) {

            indexGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t *>(index),
                indexLength_);
        }

        if (mode_ == 1 &&
            ptr != nullptr &&
            ptrLength_ > 0) {

            ptrGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t *>(ptr),
                ptrLength_);
        }


        // ----------------------------------------------------
        // UB
        //
        // calcBuf_:
        //   float expSrc[256]
        //   float expDst[256]
        //
        // posBuf_:
        //   uint32_t positions[256]
        //
        // ----------------------------------------------------

        pipe_.InitBuffer(
            calcBuf_,
            SOFTMAX_CALC_ELEMS * sizeof(float));

        pipe_.InitBuffer(
            posBuf_,
            SOFTMAX_TILE_ELEMS * sizeof(uint32_t));
    }


    // ========================================================
    // Process
    // ========================================================

    __aicore__ inline void Process()
    {
        if (length_ == 0 ||
            outerSize_ == 0 ||
            dimSize_ == 0 ||
            innerSize_ == 0) {

            return;
        }

        if (mode_ == 0) {
            ProcessIndex();
        } else {
            ProcessPtr();
        }
    }


private:

    // ========================================================
    // Common helpers
    // ========================================================

    __aicore__ inline uint32_t LinearIndex(
        uint32_t outer,
        uint32_t d,
        uint32_t inner) const
    {
        return outer * dimSize_ * innerSize_
             + d * innerSize_
             + inner;
    }


    __aicore__ inline float LoadSrc(
        uint32_t idx)
    {
        return ToFloatScalar(
            srcGm_.GetValue(idx));
    }


    __aicore__ inline void StoreOut(
        uint32_t idx,
        float value)
    {
        outGm_.SetValue(
            idx,
            FromFloatScalar<DT_SRC>(value));
    }


    // ========================================================
    // index value
    // ========================================================

    __aicore__ inline int32_t GroupId(
        uint32_t outer,
        uint32_t d,
        uint32_t inner)
    {
        // Support:
        //
        // 1. index length == dimSize
        //
        // 2. index has same number of elements as src
        //

        if (indexLength_ == length_) {

            return indexGm_.GetValue(
                LinearIndex(
                    outer,
                    d,
                    inner));
        }

        return indexGm_.GetValue(d);
    }


    // ========================================================
    // Batched vector Exp
    // ========================================================

    __aicore__ inline void RunExp(
        uint32_t count)
    {
        if (count == 0) {
            return;
        }

        LocalTensor<float> calc =
            calcBuf_.Get<float>(
                SOFTMAX_CALC_ELEMS);

        LocalTensor<float> expSrc =
            calc;

        LocalTensor<float> expDst =
            calc[SOFTMAX_TILE_ELEMS];


        // ----------------------------------------------------
        // Scalar SetValue -> Vector
        // ----------------------------------------------------

        TEventID eventSV =
            GetTPipePtr()->FetchEventID(
                HardEvent::S_V);

        SetFlag<HardEvent::S_V>(
            eventSV);

        WaitFlag<HardEvent::S_V>(
            eventSV);


        // ----------------------------------------------------
        // Vector Exp
        // ----------------------------------------------------

        AscendC::Exp(
            expDst,
            expSrc,
            count);


        // ----------------------------------------------------
        // Vector -> Scalar GetValue
        // ----------------------------------------------------

        TEventID eventVS =
            GetTPipePtr()->FetchEventID(
                HardEvent::V_S);

        SetFlag<HardEvent::V_S>(
            eventVS);

        WaitFlag<HardEvent::V_S>(
            eventVS);
    }


    // ========================================================
    // INDEX MODE
    // ========================================================

    __aicore__ inline void ProcessIndex()
    {
        if (indexLength_ == 0) {
            return;
        }


        LocalTensor<float> calc =
            calcBuf_.Get<float>(
                SOFTMAX_CALC_ELEMS);

        LocalTensor<float> expSrc =
            calc;

        LocalTensor<float> expDst =
            calc[SOFTMAX_TILE_ELEMS];

        LocalTensor<uint32_t> positions =
            posBuf_.Get<uint32_t>(
                SOFTMAX_TILE_ELEMS);


        // ====================================================
        // Multi-core:
        //
        // each (outer, inner) pair is independent.
        // ====================================================

        uint32_t blockIdx =
            GetBlockIdx();

        uint32_t blockNum =
            GetBlockNum();

        uint32_t totalSlices =
            outerSize_ * innerSize_;


        for (uint32_t slice = blockIdx;
             slice < totalSlices;
             slice += blockNum) {

            uint32_t outer =
                slice / innerSize_;

            uint32_t inner =
                slice % innerSize_;


            // =================================================
            // Find maximum group id
            // =================================================

            int32_t maxGroup = -1;

            for (uint32_t d = 0;
                 d < dimSize_;
                 ++d) {

                int32_t group =
                    GroupId(
                        outer,
                        d,
                        inner);

                if (group > maxGroup) {
                    maxGroup = group;
                }
            }

            if (maxGroup < 0) {
                continue;
            }


            uint32_t numGroups =
                static_cast<uint32_t>(
                    maxGroup + 1);


            // =================================================
            // Process every group
            // =================================================

            for (uint32_t group = 0;
                 group < numGroups;
                 ++group) {


                // =============================================
                // Pass 1:
                //
                // simultaneously:
                //   1. calculate max
                //   2. count group size
                //
                // =============================================

                float maxValue =
                    -3.402823466e38f;

                uint32_t groupSize = 0;


                for (uint32_t d = 0;
                     d < dimSize_;
                     ++d) {

                    int32_t currentGroup =
                        GroupId(
                            outer,
                            d,
                            inner);

                    if (currentGroup !=
                        static_cast<int32_t>(
                            group)) {

                        continue;
                    }


                    uint32_t idx =
                        LinearIndex(
                            outer,
                            d,
                            inner);

                    float value =
                        LoadSrc(idx);


                    if (groupSize == 0 ||
                        value > maxValue) {

                        maxValue = value;
                    }

                    ++groupSize;
                }


                if (groupSize == 0) {
                    continue;
                }


                // =================================================
                // FAST PATH
                //
                // Entire group fits in UB:
                //
                // x-max
                //   ↓
                // Exp               only ONCE
                //   ↓
                // sum
                //   ↓
                // reuse exp result
                //   ↓
                // output
                //
                // =================================================

                if (groupSize <=
                    SOFTMAX_TILE_ELEMS) {

                    uint32_t count = 0;


                    // -----------------------------------------
                    // Gather this group into UB
                    // -----------------------------------------

                    for (uint32_t d = 0;
                         d < dimSize_;
                         ++d) {

                        int32_t currentGroup =
                            GroupId(
                                outer,
                                d,
                                inner);

                        if (currentGroup !=
                            static_cast<int32_t>(
                                group)) {

                            continue;
                        }


                        uint32_t idx =
                            LinearIndex(
                                outer,
                                d,
                                inner);

                        float value =
                            LoadSrc(idx);


                        expSrc.SetValue(
                            count,
                            value - maxValue);

                        positions.SetValue(
                            count,
                            d);

                        ++count;
                    }


                    // -----------------------------------------
                    // Exp only once
                    // -----------------------------------------

                    RunExp(count);


                    // -----------------------------------------
                    // Sum
                    // -----------------------------------------

                    float sumExp = 0.0f;

                    for (uint32_t i = 0;
                         i < count;
                         ++i) {

                        sumExp +=
                            expDst.GetValue(i);
                    }


                    // Only one division per group.
                    float invDenom =
                        1.0f /
                        (sumExp + eps_);


                    // -----------------------------------------
                    // Reuse Exp result
                    // -----------------------------------------

                    for (uint32_t i = 0;
                         i < count;
                         ++i) {

                        uint32_t d =
                            positions.GetValue(i);

                        uint32_t idx =
                            LinearIndex(
                                outer,
                                d,
                                inner);

                        float result =
                            expDst.GetValue(i)
                            * invDenom;

                        StoreOut(
                            idx,
                            result);
                    }


                    continue;
                }


                // =================================================
                // FALLBACK
                //
                // groupSize > 256
                //
                // Keep tiled implementation.
                // =================================================


                // =============================================
                // Pass 2:
                // tiled Exp + sum
                // =============================================

                float sumExp = 0.0f;

                uint32_t count = 0;


                for (uint32_t d = 0;
                     d < dimSize_;
                     ++d) {

                    int32_t currentGroup =
                        GroupId(
                            outer,
                            d,
                            inner);

                    if (currentGroup !=
                        static_cast<int32_t>(
                            group)) {

                        continue;
                    }


                    uint32_t idx =
                        LinearIndex(
                            outer,
                            d,
                            inner);

                    float value =
                        LoadSrc(idx);


                    expSrc.SetValue(
                        count,
                        value - maxValue);

                    ++count;


                    if (count ==
                        SOFTMAX_TILE_ELEMS) {

                        RunExp(count);


                        for (uint32_t i = 0;
                             i < count;
                             ++i) {

                            sumExp +=
                                expDst.GetValue(i);
                        }

                        count = 0;
                    }
                }


                if (count > 0) {

                    RunExp(count);


                    for (uint32_t i = 0;
                         i < count;
                         ++i) {

                        sumExp +=
                            expDst.GetValue(i);
                    }
                }


                float invDenom =
                    1.0f /
                    (sumExp + eps_);


                // =============================================
                // Pass 3:
                // tiled Exp + normalize
                // =============================================

                count = 0;


                for (uint32_t d = 0;
                     d < dimSize_;
                     ++d) {

                    int32_t currentGroup =
                        GroupId(
                            outer,
                            d,
                            inner);

                    if (currentGroup !=
                        static_cast<int32_t>(
                            group)) {

                        continue;
                    }


                    uint32_t idx =
                        LinearIndex(
                            outer,
                            d,
                            inner);

                    float value =
                        LoadSrc(idx);


                    expSrc.SetValue(
                        count,
                        value - maxValue);

                    positions.SetValue(
                        count,
                        d);

                    ++count;


                    if (count ==
                        SOFTMAX_TILE_ELEMS) {

                        RunExp(count);


                        for (uint32_t i = 0;
                             i < count;
                             ++i) {

                            uint32_t savedD =
                                positions.GetValue(i);

                            uint32_t outIdx =
                                LinearIndex(
                                    outer,
                                    savedD,
                                    inner);

                            float result =
                                expDst.GetValue(i)
                                * invDenom;

                            StoreOut(
                                outIdx,
                                result);
                        }


                        count = 0;
                    }
                }


                if (count > 0) {

                    RunExp(count);


                    for (uint32_t i = 0;
                         i < count;
                         ++i) {

                        uint32_t savedD =
                            positions.GetValue(i);

                        uint32_t outIdx =
                            LinearIndex(
                                outer,
                                savedD,
                                inner);

                        float result =
                            expDst.GetValue(i)
                            * invDenom;

                        StoreOut(
                            outIdx,
                            result);
                    }
                }
            }
        }
    }


    // ========================================================
    // PTR / CSR MODE
    // ========================================================

    __aicore__ inline void ProcessPtr()
    {
        if (ptrLength_ < 2) {
            return;
        }


        LocalTensor<float> calc =
            calcBuf_.Get<float>(
                SOFTMAX_CALC_ELEMS);

        LocalTensor<float> expSrc =
            calc;

        LocalTensor<float> expDst =
            calc[SOFTMAX_TILE_ELEMS];


        uint32_t numGroups =
            ptrLength_ - 1;


        // ====================================================
        // Multi-core split
        // ====================================================

        uint32_t blockIdx =
            GetBlockIdx();

        uint32_t blockNum =
            GetBlockNum();

        uint32_t totalSlices =
            outerSize_ * innerSize_;


        for (uint32_t slice = blockIdx;
             slice < totalSlices;
             slice += blockNum) {

            uint32_t outer =
                slice / innerSize_;

            uint32_t inner =
                slice % innerSize_;


            // =================================================
            // Every CSR group
            // =================================================

            for (uint32_t group = 0;
                 group < numGroups;
                 ++group) {

                int32_t start =
                    ptrGm_.GetValue(group);

                int32_t end =
                    ptrGm_.GetValue(
                        group + 1);


                // ---------------------------------------------
                // Validate
                // ---------------------------------------------

                if (start < 0 ||
                    end < start ||
                    end >
                    static_cast<int32_t>(
                        dimSize_)) {

                    continue;
                }


                if (start == end) {
                    continue;
                }


                uint32_t groupSize =
                    static_cast<uint32_t>(
                        end - start);


                // =============================================
                // Pass 1:
                // max
                // =============================================

                float maxValue =
                    -3.402823466e38f;


                for (int32_t d = start;
                     d < end;
                     ++d) {

                    uint32_t idx =
                        LinearIndex(
                            outer,
                            static_cast<uint32_t>(
                                d),
                            inner);

                    float value =
                        LoadSrc(idx);


                    if (value > maxValue) {
                        maxValue = value;
                    }
                }


                // =================================================
                // FAST PATH
                //
                // group <= 256
                //
                // Only one Exp.
                // =================================================

                if (groupSize <=
                    SOFTMAX_TILE_ELEMS) {


                    // -----------------------------------------
                    // Gather x-max
                    // -----------------------------------------

                    for (uint32_t i = 0;
                         i < groupSize;
                         ++i) {

                        uint32_t d =
                            static_cast<uint32_t>(
                                start)
                            + i;

                        uint32_t idx =
                            LinearIndex(
                                outer,
                                d,
                                inner);

                        float value =
                            LoadSrc(idx);


                        expSrc.SetValue(
                            i,
                            value - maxValue);
                    }


                    // -----------------------------------------
                    // Exp only once
                    // -----------------------------------------

                    RunExp(groupSize);


                    // -----------------------------------------
                    // Sum
                    // -----------------------------------------

                    float sumExp = 0.0f;


                    for (uint32_t i = 0;
                         i < groupSize;
                         ++i) {

                        sumExp +=
                            expDst.GetValue(i);
                    }


                    float invDenom =
                        1.0f /
                        (sumExp + eps_);


                    // -----------------------------------------
                    // Normalize
                    //
                    // reuse expDst directly.
                    // -----------------------------------------

                    for (uint32_t i = 0;
                         i < groupSize;
                         ++i) {

                        uint32_t d =
                            static_cast<uint32_t>(
                                start)
                            + i;

                        uint32_t idx =
                            LinearIndex(
                                outer,
                                d,
                                inner);

                        float result =
                            expDst.GetValue(i)
                            * invDenom;

                        StoreOut(
                            idx,
                            result);
                    }


                    continue;
                }


                // =================================================
                // FALLBACK
                //
                // groupSize > 256
                // =================================================


                // =============================================
                // Pass 2:
                // tiled Exp + sum
                // =============================================

                float sumExp = 0.0f;

                int32_t d =
                    start;


                while (d < end) {

                    uint32_t count = 0;


                    while (d < end &&
                           count <
                           SOFTMAX_TILE_ELEMS) {

                        uint32_t idx =
                            LinearIndex(
                                outer,
                                static_cast<uint32_t>(
                                    d),
                                inner);

                        float value =
                            LoadSrc(idx);


                        expSrc.SetValue(
                            count,
                            value - maxValue);

                        ++count;
                        ++d;
                    }


                    RunExp(count);


                    for (uint32_t i = 0;
                         i < count;
                         ++i) {

                        sumExp +=
                            expDst.GetValue(i);
                    }
                }


                float invDenom =
                    1.0f /
                    (sumExp + eps_);


                // =============================================
                // Pass 3:
                // tiled Exp + normalization
                // =============================================

                d = start;


                while (d < end) {

                    uint32_t count = 0;

                    int32_t tileStart =
                        d;


                    while (d < end &&
                           count <
                           SOFTMAX_TILE_ELEMS) {

                        uint32_t idx =
                            LinearIndex(
                                outer,
                                static_cast<uint32_t>(
                                    d),
                                inner);

                        float value =
                            LoadSrc(idx);


                        expSrc.SetValue(
                            count,
                            value - maxValue);

                        ++count;
                        ++d;
                    }


                    RunExp(count);


                    for (uint32_t i = 0;
                         i < count;
                         ++i) {

                        uint32_t currentD =
                            static_cast<uint32_t>(
                                tileStart +
                                static_cast<int32_t>(
                                    i));

                        uint32_t idx =
                            LinearIndex(
                                outer,
                                currentD,
                                inner);

                        float result =
                            expDst.GetValue(i)
                            * invDenom;

                        StoreOut(
                            idx,
                            result);
                    }
                }
            }
        }
    }


private:

    // ========================================================
    // Global memory
    // ========================================================

    GlobalTensor<DT_SRC> srcGm_;

    GlobalTensor<DT_SRC> outGm_;

    GlobalTensor<int32_t> indexGm_;

    GlobalTensor<int32_t> ptrGm_;


    // ========================================================
    // Local memory
    // ========================================================

    TPipe pipe_;

    TBuf<TPosition::VECCALC> calcBuf_;

    TBuf<TPosition::VECCALC> posBuf_;


    // ========================================================
    // Tiling
    // ========================================================

    uint32_t length_ = 0;

    uint32_t mode_ = 0;

    int32_t dim_ = 0;

    uint32_t outerSize_ = 1;

    uint32_t dimSize_ = 1;

    uint32_t innerSize_ = 1;

    uint32_t ptrLength_ = 0;

    uint32_t indexLength_ = 0;

    float eps_ = 1e-16f;
};


// ============================================================
// Kernel entry
// ============================================================

template <typename DT_SRC>
__global__ __aicore__ void sparse_softmax(
    GM_ADDR src,
    GM_ADDR index,
    GM_ADDR ptr,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        SparseSoftmaxTilingData);


    GET_TILING_DATA_WITH_STRUCT(
        SparseSoftmaxTilingData,
        tiling_data,
        tiling);


    KernelSparseSoftmax<DT_SRC> op;


    op.Init(
        src,
        index,
        ptr,
        out,
        tiling_data);


    op.Process();
}