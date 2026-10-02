// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_pre_tiling.h"
#include "../op_kernel/tiling_key_mhc_pre.h"

namespace optiling {
    static ge::graphStatus TilingFunc(
        gert::TilingContext *context)
    {
        // 原模板：平台信息
        auto platform =
            platform_ascendc::PlatformAscendC(
                context->GetPlatformInfo());

        int32_t num_cores_aiv =
            platform.GetCoreNumAiv();

        uint64_t ub_size;
        platform.GetCoreMemSize(
            platform_ascendc::CoreMemType::UB,
            ub_size);

        // 原模板：输入
        const gert::Tensor *tensor_x =
            context->GetRequiredInputTensor(0);

        const gert::Tensor *tensor_phi =
            context->GetRequiredInputTensor(1);

        const gert::Tensor *tensor_alpha =
            context->GetRequiredInputTensor(2);

        const gert::Tensor *tensor_bias =
            context->GetRequiredInputTensor(3);

        const gert::Tensor *tensor_gamma =
            context->GetOptionalInputTensor(4);

        // 原模板：dtype / size
        ge::DataType dtype_x =
            tensor_x->GetDataType();

        int dtype_size_x =
            ge::GetSizeByDataType(dtype_x);

        uint32_t length_x =
            tensor_x->GetShapeSize();

        uint32_t size_x =
            tensor_x->GetSize();

        // 新增：x = (B,S,n,D)
        const gert::Shape &x_shape =
            tensor_x->GetStorageShape();

        uint32_t B =
            static_cast<uint32_t>(
                x_shape.GetDim(0));

        uint32_t S =
            static_cast<uint32_t>(
                x_shape.GetDim(1));

        uint32_t n =
            static_cast<uint32_t>(
                x_shape.GetDim(2));

        uint32_t D =
            static_cast<uint32_t>(
                x_shape.GetDim(3));

        uint32_t row_num =
            B * S;

        uint32_t nD =
            n * D;

        uint32_t proj_dim =
            n * n + 2 * n;

        // uint32_t used_core_num =
        //     row_num <
        //     static_cast<uint32_t>(num_cores_aiv)
        //         ? row_num
        //         : static_cast<uint32_t>(
        //             num_cores_aiv);

        // uint32_t rows_per_core =
        //     (row_num + used_core_num - 1)
        //     / used_core_num;
        uint32_t used_core_num = 1;
        uint32_t rows_per_core = row_num;
        // 原模板：属性
        const gert::RuntimeAttrs *attrs =
            context->GetAttrs();

        const float *attr_normEps =
            attrs->GetFloat(0);

        const float *attr_hcEps =
            attrs->GetFloat(1);

        // 原模板：dtype tiling key
        uint32_t DT_X =
            static_cast<uint32_t>(dtype_x);

        ASCENDC_TPL_SEL_PARAM(
            context,
            DT_X);

        // tiling
        MhcPreTilingData *tiling =
            context->GetTilingData<
                MhcPreTilingData>();

        // 原模板字段
        tiling->length =
            length_x;

        // 新增字段
        tiling->batchSize =
            B;

        tiling->seqLen =
            S;

        tiling->n =
            n;

        tiling->hiddenDim =
            D;

        tiling->rowNum =
            row_num;

        tiling->nD =
            nD;

        tiling->projDim =
            proj_dim;

        tiling->usedCoreNum =
            used_core_num;

        tiling->rowsPerCore =
            rows_per_core;

        tiling->hasGamma =
            tensor_gamma != nullptr
                ? 1
                : 0;

        tiling->normEps =
            attr_normEps != nullptr
                ? *attr_normEps
                : 1e-6f;

        tiling->hcEps =
            attr_hcEps != nullptr
                ? *attr_hcEps
                : 1e-6f;
        
        tiling->invND =
            1.0f / static_cast<float>(nD);

        // 启动核数
        context->SetBlockDim(
            used_core_num);

        // workspace
        size_t *currentWorkspace =
            context->GetWorkspaceSizes(1);

        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const gert::Shape *xShape =
        context->GetInputShape(0);

    if (xShape == nullptr) {
        return GRAPH_FAILED;
    }

    int64_t B =
        xShape->GetDim(0);

    int64_t S =
        xShape->GetDim(1);

    int64_t n =
        xShape->GetDim(2);

    int64_t D =
        xShape->GetDim(3);

    // hIn: [B, S, D]
    gert::Shape *hInShape =
        context->GetOutputShape(0);

    hInShape->SetDimNum(3);
    hInShape->SetDim(0, B);
    hInShape->SetDim(1, S);
    hInShape->SetDim(2, D);

    // hPost: [B, S, n]
    gert::Shape *hPostShape =
        context->GetOutputShape(1);

    hPostShape->SetDimNum(3);
    hPostShape->SetDim(0, B);
    hPostShape->SetDim(1, S);
    hPostShape->SetDim(2, n);

    // hRes: [B, S, n, n]
    gert::Shape *hResShape =
        context->GetOutputShape(2);

    hResShape->SetDimNum(4);
    hResShape->SetDim(0, B);
    hResShape->SetDim(1, S);
    hResShape->SetDim(2, n);
    hResShape->SetDim(3, n);

    return GRAPH_SUCCESS;
}

    static graphStatus InferDataType(
        gert::InferDataTypeContext *context)
    {
        ge::DataType xDtype =
            context->GetInputDataType(0);

        // hIn 跟 x
        context->SetOutputDataType(
            0,
            xDtype);

        // hPost 固定 FP32
        context->SetOutputDataType(
            1,
            ge::DT_FLOAT);

        // hRes 固定 FP32
        context->SetOutputDataType(
            2,
            ge::DT_FLOAT);

        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class MhcPre : public OpDef {
    public:
        explicit MhcPre(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});

            this->Input("phi")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});

            this->Input("alpha")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});

            this->Input("bias")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});

            this->Input("gamma")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hIn")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hPost")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("hRes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("normEps").AttrType(OPTIONAL).Float(1e-06);
            this->Attr("hcEps").AttrType(OPTIONAL).Float(1e-06);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcPre);
}  // namespace ops
