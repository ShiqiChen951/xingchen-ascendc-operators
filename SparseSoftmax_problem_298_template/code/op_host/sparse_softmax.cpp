// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/sparse_softmax_tiling.h"
#include "../op_kernel/tiling_key_sparse_softmax.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_src = context->GetRequiredInputTensor(0);
        const auto &srcShape = tensor_src->GetStorageShape();
        int32_t rank = static_cast<int32_t>(srcShape.GetDimNum());
        const gert::Tensor *tensor_index = context->GetOptionalInputTensor(1);
        const gert::Tensor *tensor_ptr = context->GetOptionalInputTensor(2);

        uint32_t indexLength = 0;
        if (tensor_index != nullptr) {
            indexLength =
                static_cast<uint32_t>(
                    tensor_index->GetShapeSize());
        }
        uint32_t ptrLength = 0;
        if (tensor_ptr != nullptr) {
            ptrLength =
                static_cast<uint32_t>(tensor_ptr->GetShapeSize());
        }

        uint32_t mode = 0;

        if (indexLength > 0) {
            mode = 0;  // index 模式
        } else if (ptrLength > 0) {
            mode = 1;  // ptr 模式
        }
        ge::DataType dtype_src = tensor_src->GetDataType(); // 获取数据类型
        int dtype_size_src = ge::GetSizeByDataType(dtype_src); // 获取数据类型的字长
        uint32_t length_src = tensor_src->GetShapeSize(); // 获取元素个数
        uint32_t size_src = tensor_src->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_dim = attrs->GetInt(0);
        const float *attr_eps = attrs->GetFloat(1);
        float eps = 1e-16f;

        if (attr_eps != nullptr) {
            eps = *attr_eps;
        }
        int32_t dim = 0;

        if (attr_dim != nullptr) {
            dim = static_cast<int32_t>(*attr_dim);
        }
        if (dim < 0) {
            dim += rank;
        }

        uint32_t outerSize = 1;
        uint32_t dimSize = 1;
        uint32_t innerSize = 1;

        for (int32_t i = 0; i < dim; ++i) {
            outerSize *= static_cast<uint32_t>(srcShape.GetDim(i));
        }

        dimSize = static_cast<uint32_t>(srcShape.GetDim(dim));

        for (int32_t i = dim + 1; i < rank; ++i) {
            innerSize *= static_cast<uint32_t>(srcShape.GetDim(i));
        }
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_SRC = static_cast<uint32_t>(dtype_src);
        ASCENDC_TPL_SEL_PARAM(context, DT_SRC);
        // 示例: 计算tiling方案并填充tiling结构体
        SparseSoftmaxTilingData *tiling = context->GetTilingData<SparseSoftmaxTilingData>();

        tiling->length = length_src;
        tiling->mode = mode;
        tiling->dim = dim;

        tiling->outerSize = outerSize;
        tiling->dimSize = dimSize;
        tiling->innerSize = innerSize;

        tiling->ptrLength = ptrLength;
        tiling->indexLength = indexLength;
        tiling->eps = eps;
        // 配置启动核数
        // context->SetBlockDim(num_cores_aiv);
        context->SetBlockDim(1);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = platform.GetLibApiWorkSpaceSize();
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *srcShape = context->GetInputShape(0);
        gert::Shape *outShape = context->GetOutputShape(0);

        *outShape = *srcShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto srcDataType = context->GetInputDataType(0);

        context->SetOutputDataType(0, srcDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class SparseSoftmax : public OpDef {
    public:
        explicit SparseSoftmax(const char *name) : OpDef(name) {
            this->Input("src")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("index")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("ptr")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("dim").AttrType(OPTIONAL).Int(0);
            this->Attr("eps").AttrType(OPTIONAL).Float(1e-16);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SparseSoftmax);
}  // namespace ops
