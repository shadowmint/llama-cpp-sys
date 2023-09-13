#import "ggml-metal.h"

#import "ggml.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// TODO: temporary - reuse llama.cpp logging
#ifdef GGML_METAL_NDEBUG
#define metal_printf(...)
#else
#define metal_printf(...) fprintf(stderr, __VA_ARGS__)
#endif

#define UNUSED(x) (void)(x)

#define GGML_MAX_CONCUR (2*GGML_MAX_NODES)

struct ggml_metal_buffer {
    const char * name;

    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_context {
    int n_cb;

    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    id<MTLCommandBuffer>         command_buffers [GGML_METAL_MAX_COMMAND_BUFFERS];
    id<MTLComputeCommandEncoder> command_encoders[GGML_METAL_MAX_COMMAND_BUFFERS];

    dispatch_queue_t d_queue;

    int n_buffers;
    struct ggml_metal_buffer buffers[GGML_METAL_MAX_BUFFERS];

    int concur_list[GGML_MAX_CONCUR];
    int concur_list_len;

    // custom kernels
#define GGML_METAL_DECL_KERNEL(name) \
    id<MTLFunction>             function_##name; \
    id<MTLComputePipelineState> pipeline_##name

    GGML_METAL_DECL_KERNEL(add);
    GGML_METAL_DECL_KERNEL(add_row); // TODO: avoid this extra kernel, instead extend the "add" kernel to support broadcast
    GGML_METAL_DECL_KERNEL(mul);
    GGML_METAL_DECL_KERNEL(mul_row); // TODO: avoid this extra kernel, instead extend the "mul" kernel to support broadcast
    GGML_METAL_DECL_KERNEL(scale);
    GGML_METAL_DECL_KERNEL(silu);
    GGML_METAL_DECL_KERNEL(relu);
    GGML_METAL_DECL_KERNEL(gelu);
    GGML_METAL_DECL_KERNEL(soft_max);
    GGML_METAL_DECL_KERNEL(soft_max_4);
    GGML_METAL_DECL_KERNEL(diag_mask_inf);
    GGML_METAL_DECL_KERNEL(diag_mask_inf_8);
    GGML_METAL_DECL_KERNEL(get_rows_f16);
    GGML_METAL_DECL_KERNEL(get_rows_q4_0);
    GGML_METAL_DECL_KERNEL(get_rows_q4_1);
    GGML_METAL_DECL_KERNEL(get_rows_q8_0);
    GGML_METAL_DECL_KERNEL(get_rows_q2_K);
    GGML_METAL_DECL_KERNEL(get_rows_q3_K);
    GGML_METAL_DECL_KERNEL(get_rows_q4_K);
    GGML_METAL_DECL_KERNEL(get_rows_q5_K);
    GGML_METAL_DECL_KERNEL(get_rows_q6_K);
    GGML_METAL_DECL_KERNEL(rms_norm);
    GGML_METAL_DECL_KERNEL(norm);
    GGML_METAL_DECL_KERNEL(mul_mat_f16_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_f16_f32_1row);
    GGML_METAL_DECL_KERNEL(mul_mat_f16_f32_l4);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_0_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_1_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q8_0_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q2_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q3_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q5_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q6_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_f16_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_q4_0_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_q4_1_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_q8_0_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_q2_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_q3_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_q4_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_q5_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mm_q6_K_f32);
    GGML_METAL_DECL_KERNEL(rope);
    GGML_METAL_DECL_KERNEL(alibi_f32);
    GGML_METAL_DECL_KERNEL(cpy_f32_f16);
    GGML_METAL_DECL_KERNEL(cpy_f32_f32);
    GGML_METAL_DECL_KERNEL(cpy_f16_f16);

#undef GGML_METAL_DECL_KERNEL
};

// MSL code
// TODO: move the contents here when ready
//       for now it is easier to work in a separate file
static NSString * const msl_library_source = @"#include <metal_stdlib>\n"
                                             "\n"
                                             "using namespace metal;\n"
                                             "\n"
                                             "#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
                                             "\n"
                                             "#define QK4_0 32\n"
                                             "#define QR4_0 2\n"
                                             "typedef struct {\n"
                                             "    half    d;             // delta\n"
                                             "    uint8_t qs[QK4_0 / 2]; // nibbles / quants\n"
                                             "} block_q4_0;\n"
                                             "\n"
                                             "#define QK4_1 32\n"
                                             "typedef struct {\n"
                                             "    half d;          // delta\n"
                                             "    half m;          // min\n"
                                             "    uint8_t qs[QK4_1 / 2];  // nibbles / quants\n"
                                             "} block_q4_1;\n"
                                             "\n"
                                             "#define QK8_0 32\n"
                                             "typedef struct {\n"
                                             "    half    d;         // delta\n"
                                             "    int8_t  qs[QK8_0]; // quants\n"
                                             "} block_q8_0;\n"
                                             "\n"
                                             "kernel void kernel_add(\n"
                                             "        device const float4 * src0,\n"
                                             "        device const float4 * src1,\n"
                                             "        device       float4 * dst,\n"
                                             "        uint tpig[[thread_position_in_grid]]) {\n"
                                             "    dst[tpig] = src0[tpig] + src1[tpig];\n"
                                             "}\n"
                                             "\n"
                                             "// assumption: src1 is a row\n"
                                             "// broadcast src1 into src0\n"
                                             "kernel void kernel_add_row(\n"
                                             "        device const float4 * src0,\n"
                                             "        device const float4 * src1,\n"
                                             "        device       float4 * dst,\n"
                                             "        constant   int64_t & nb,\n"
                                             "        uint tpig[[thread_position_in_grid]]) {\n"
                                             "    dst[tpig] = src0[tpig] + src1[tpig % nb];\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_mul(\n"
                                             "        device const float4 * src0,\n"
                                             "        device const float4 * src1,\n"
                                             "        device       float4 * dst,\n"
                                             "        uint tpig[[thread_position_in_grid]]) {\n"
                                             "    dst[tpig] = src0[tpig] * src1[tpig];\n"
                                             "}\n"
                                             "\n"
                                             "// assumption: src1 is a row\n"
                                             "// broadcast src1 into src0\n"
                                             "kernel void kernel_mul_row(\n"
                                             "        device const float4 * src0,\n"
                                             "        device const float4 * src1,\n"
                                             "        device       float4 * dst,\n"
                                             "        constant    int64_t & nb,\n"
                                             "        uint tpig[[thread_position_in_grid]]) {\n"
                                             "    dst[tpig] = src0[tpig] * src1[tpig % nb];\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_scale(\n"
                                             "        device const float4 * src0,\n"
                                             "        device       float4 * dst,\n"
                                             "        constant     float & scale,\n"
                                             "        uint tpig[[thread_position_in_grid]]) {\n"
                                             "    dst[tpig] = src0[tpig] * scale;\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_silu(\n"
                                             "        device const float4 * src0,\n"
                                             "        device       float4 * dst,\n"
                                             "        uint tpig[[thread_position_in_grid]]) {\n"
                                             "    device const float4 & x = src0[tpig];\n"
                                             "    dst[tpig] = x / (1.0f + exp(-x));\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_relu(\n"
                                             "        device const float * src0,\n"
                                             "        device       float * dst,\n"
                                             "        uint tpig[[thread_position_in_grid]]) {\n"
                                             "    dst[tpig] = max(0.0f, src0[tpig]);\n"
                                             "}\n"
                                             "\n"
                                             "constant float GELU_COEF_A    = 0.044715f;\n"
                                             "constant float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;\n"
                                             "\n"
                                             "kernel void kernel_gelu(\n"
                                             "    device const float4 * src0,\n"
                                             "    device       float4 * dst,\n"
                                             "    uint tpig[[thread_position_in_grid]]) {\n"
                                             "    device const float4 & x = src0[tpig];\n"
                                             "\n"
                                             "    // BEWARE !!!\n"
                                             "    // Simply using \"tanh\" instead of \"precise::tanh\" will sometimes results in NaNs!\n"
                                             "    // This was observed with Falcon 7B and 40B models\n"
                                             "    //\n"
                                             "    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_soft_max(\n"
                                             "        device const float * src0,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint3 tpitg[[thread_position_in_threadgroup]],\n"
                                             "        uint3   ntg[[threads_per_threadgroup]]) {\n"
                                             "    const int64_t i03 = tgpig[2];\n"
                                             "    const int64_t i02 = tgpig[1];\n"
                                             "    const int64_t i01 = tgpig[0];\n"
                                             "\n"
                                             "    device const float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n"
                                             "    device       float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n"
                                             "\n"
                                             "    // parallel max\n"
                                             "    float lmax = psrc0[tpitg[0]];\n"
                                             "    for (int i00 = tpitg[0] + ntg[0]; i00 < ne00; i00 += ntg[0]) {\n"
                                             "        lmax = MAX(lmax, psrc0[i00]);\n"
                                             "    }\n"
                                             "    const float max = simd_max(lmax);\n"
                                             "\n"
                                             "    // parallel sum\n"
                                             "    float lsum = 0.0f;\n"
                                             "    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {\n"
                                             "        const float exp_psrc0 = exp(psrc0[i00] - max);\n"
                                             "        lsum += exp_psrc0;\n"
                                             "        // Remember the result of exp here. exp is expensive, so we really do not\n"
                                             "        // whish to compute it twice.\n"
                                             "        pdst[i00] = exp_psrc0;\n"
                                             "    }\n"
                                             "\n"
                                             "    const float sum = simd_sum(lsum);\n"
                                             "\n"
                                             "    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {\n"
                                             "        pdst[i00] /= sum;\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_soft_max_4(\n"
                                             "        device const float * src0,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint3 tpitg[[thread_position_in_threadgroup]],\n"
                                             "        uint3   ntg[[threads_per_threadgroup]]) {\n"
                                             "    const int64_t i03 = tgpig[2];\n"
                                             "    const int64_t i02 = tgpig[1];\n"
                                             "    const int64_t i01 = tgpig[0];\n"
                                             "\n"
                                             "    device const float4 * psrc4 = (device const float4 *)(src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);\n"
                                             "    device       float4 * pdst4 = (device       float4 *)(dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);\n"
                                             "\n"
                                             "    // parallel max\n"
                                             "    float4 lmax4 = psrc4[tpitg[0]];\n"
                                             "    for (int i00 = tpitg[0] + ntg[0]; i00 < ne00/4; i00 += ntg[0]) {\n"
                                             "        lmax4 = fmax(lmax4, psrc4[i00]);\n"
                                             "    }\n"
                                             "    float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));\n"
                                             "\n"
                                             "    const float max = simd_max(lmax);\n"
                                             "\n"
                                             "    // parallel sum\n"
                                             "    float4 lsum4 = 0.0f;\n"
                                             "    for (int i00 = tpitg[0]; i00 < ne00/4; i00 += ntg[0]) {\n"
                                             "        const float4 exp_psrc4 = exp(psrc4[i00] - max);\n"
                                             "        lsum4 += exp_psrc4;\n"
                                             "        pdst4[i00] = exp_psrc4;\n"
                                             "    }\n"
                                             "    float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];\n"
                                             "\n"
                                             "    const float sum = simd_sum(lsum);\n"
                                             "\n"
                                             "    for (int i00 = tpitg[0]; i00 < ne00/4; i00 += ntg[0]) {\n"
                                             "        pdst4[i00] /= sum;\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_diag_mask_inf(\n"
                                             "        device const float * src0,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant       int & n_past,\n"
                                             "        uint3 tpig[[thread_position_in_grid]]) {\n"
                                             "    const int64_t i02 = tpig[2];\n"
                                             "    const int64_t i01 = tpig[1];\n"
                                             "    const int64_t i00 = tpig[0];\n"
                                             "\n"
                                             "    if (i00 > n_past + i01) {\n"
                                             "        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;\n"
                                             "    } else {\n"
                                             "        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];\n"
                                             "     }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_diag_mask_inf_8(\n"
                                             "        device const float4 * src0,\n"
                                             "        device       float4 * dst,\n"
                                             "        constant    int64_t & ne00,\n"
                                             "        constant    int64_t & ne01,\n"
                                             "        constant        int & n_past,\n"
                                             "        uint3 tpig[[thread_position_in_grid]]) {\n"
                                             "\n"
                                             "    const int64_t i = 2*tpig[0];\n"
                                             "\n"
                                             "    dst[i+0] = src0[i+0];\n"
                                             "    dst[i+1] = src0[i+1];\n"
                                             "    int64_t i4 = 4*i;\n"
                                             "    const int64_t i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;\n"
                                             "    const int64_t i01 = i4/(ne00);      i4 -= i01*ne00;\n"
                                             "    const int64_t i00 = i4;\n"
                                             "    for (int k = 3; k >= 0; --k) {\n"
                                             "        if (i00 + 4 + k <= n_past + i01) {\n"
                                             "            break;\n"
                                             "        }\n"
                                             "        dst[i+1][k] = -INFINITY;\n"
                                             "        if (i00 + k > n_past + i01) {\n"
                                             "            dst[i][k] = -INFINITY;\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_norm(\n"
                                             "        device const  void * src0,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant     float & eps,\n"
                                             "        threadgroup float  * sum [[threadgroup(0)]],\n"
                                             "        uint tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tpitg[[thread_position_in_threadgroup]],\n"
                                             "        uint   ntg[[threads_per_threadgroup]]) {\n"
                                             "    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);\n"
                                             "    // MEAN\n"
                                             "    // parallel sum\n"
                                             "    sum[tpitg] = 0.0f;\n"
                                             "    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {\n"
                                             "        sum[tpitg] += x[i00];\n"
                                             "    }\n"
                                             "    // reduce\n"
                                             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "    for (uint i = ntg/2; i > 0; i /= 2) {\n"
                                             "        if (tpitg < i) {\n"
                                             "            sum[tpitg] += sum[tpitg + i];\n"
                                             "        }\n"
                                             "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "    }\n"
                                             "    const float mean  = sum[0] / ne00;\n"
                                             "\n"
                                             "    // recenter and VARIANCE\n"
                                             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "    device float * y = dst + tgpig*ne00;\n"
                                             "    sum[tpitg] = 0.0f;\n"
                                             "    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {\n"
                                             "        y[i00] = x[i00] - mean;\n"
                                             "        sum[tpitg] += y[i00] * y[i00];\n"
                                             "    }\n"
                                             "\n"
                                             "    // reduce\n"
                                             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "    for (uint i = ntg/2; i > 0; i /= 2) {\n"
                                             "        if (tpitg < i) {\n"
                                             "            sum[tpitg] += sum[tpitg + i];\n"
                                             "        }\n"
                                             "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "    }\n"
                                             "    const float variance = sum[0] / ne00;\n"
                                             "\n"
                                             "    const float scale = 1.0f/sqrt(variance + eps);\n"
                                             "    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {\n"
                                             "        y[i00] = y[i00] * scale;\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_rms_norm(\n"
                                             "        device const  void * src0,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant     float & eps,\n"
                                             "        threadgroup float  * sum [[threadgroup(0)]],\n"
                                             "        uint tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tpitg[[thread_position_in_threadgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint   ntg[[threads_per_threadgroup]]) {\n"
                                             "    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);\n"
                                             "    device const float * x_scalar = (device const float *) x;\n"
                                             "    float4 sumf=0;\n"
                                             "    float all_sum=0;\n"
                                             "\n"
                                             "    // parallel sum\n"
                                             "    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {\n"
                                             "        sumf += x[i00] * x[i00];\n"
                                             "    }\n"
                                             "    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];\n"
                                             "    all_sum = simd_sum(all_sum);\n"
                                             "    if (tiisg == 0) {\n"
                                             "        sum[sgitg] = all_sum;\n"
                                             "    }\n"
                                             "\n"
                                             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "    // broadcast, simd group number is ntg / 32\n"
                                             "    for (uint i = ntg / 32 / 2; i > 0; i /= 2) {\n"
                                             "       if (tpitg < i) {\n"
                                             "           sum[tpitg] += sum[tpitg + i];\n"
                                             "       }\n"
                                             "    }\n"
                                             "    if (tpitg == 0) {\n"
                                             "        for (int i = 4 * (ne00 / 4); i < ne00; i++) {sum[0] += x_scalar[i];}\n"
                                             "        sum[0] /= ne00;\n"
                                             "    }\n"
                                             "\n"
                                             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "\n"
                                             "    const float mean  = sum[0];\n"
                                             "    const float scale = 1.0f/sqrt(mean + eps);\n"
                                             "\n"
                                             "    device float4 * y = (device float4 *) (dst + tgpig*ne00);\n"
                                             "    device float * y_scalar = (device float *) y;\n"
                                             "    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {\n"
                                             "        y[i00] = x[i00] * scale;\n"
                                             "    }\n"
                                             "    if (tpitg == 0) {\n"
                                             "        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {y_scalar[i00] = x_scalar[i00] * scale;}\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])\n"
                                             "// il indicates where the q4 quants begin (0 or QK4_0/4)\n"
                                             "// we assume that the yl's have been multiplied with the appropriate scale factor\n"
                                             "// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)\n"
                                             "inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {\n"
                                             "    float d = qb_curr->d;\n"
                                             "    float2 acc = 0.f;\n"
                                             "    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 1 + il/2);\n"
                                             "    for (int i = 0; i < 8; i+=2) {\n"
                                             "        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)\n"
                                             "                + yl[i + 1] * (qs[i / 2] & 0x0F00);\n"
                                             "        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)\n"
                                             "                + yl[i + 9] * (qs[i / 2] & 0xF000);\n"
                                             "    }\n"
                                             "    return d * (sumy * -8.f + acc[0] + acc[1]);\n"
                                             "}\n"
                                             "\n"
                                             "// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])\n"
                                             "// il indicates where the q4 quants begin (0 or QK4_0/4)\n"
                                             "// we assume that the yl's have been multiplied with the appropriate scale factor\n"
                                             "// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)\n"
                                             "inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {\n"
                                             "    float d = qb_curr->d;\n"
                                             "    float m = qb_curr->m;\n"
                                             "    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 2 + il/2);\n"
                                             "    float2 acc = 0.f;\n"
                                             "    for (int i = 0; i < 8; i+=2) {\n"
                                             "        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)\n"
                                             "                + yl[i + 1] * (qs[i / 2] & 0x0F00);\n"
                                             "        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)\n"
                                             "                + yl[i + 9] * (qs[i / 2] & 0xF000);\n"
                                             "    }\n"
                                             "    return d * (acc[0] + acc[1]) + sumy * m;\n"
                                             "}\n"
                                             "\n"
                                             "// putting them in the kernel cause a significant performance penalty\n"
                                             "#define N_DST 4 // each SIMD group works on 4 rows\n"
                                             "#define N_SIMDGROUP 2 // number of SIMD groups in a thread group\n"
                                             "#define N_SIMDWIDTH 32 // assuming SIMD group size is 32\n"
                                             "//Note: This is a template, but strictly speaking it only applies to\n"
                                             "//      quantizations where the block size is 32. It also does not\n"
                                             "//      giard against the number of rows not being divisible by\n"
                                             "//      N_DST, so this is another explicit assumption of the implementation.\n"
                                             "template<typename block_q_type, int nr, int nsg, int nw>\n"
                                             "void mul_vec_q_n_f32(device const void * src0, device const float * src1, device float * dst,\n"
                                             "                    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne10, int64_t ne12, int64_t ne0, int64_t ne1, uint gqa,\n"
                                             "                    uint3 tgpig, uint tiisg, uint sgitg) {\n"
                                             "    const int nb = ne00/QK4_0;\n"
                                             "    const int r0 = tgpig.x;\n"
                                             "    const int r1 = tgpig.y;\n"
                                             "    const int im = tgpig.z;\n"
                                             "    const int first_row = (r0 * nsg + sgitg) * nr;\n"
                                             "    const uint offset0 = first_row * nb + im/gqa*(nb*ne0);\n"
                                             "    device const block_q_type * x = (device const block_q_type *) src0 + offset0;\n"
                                             "    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;\n"
                                             "    float yl[16];       // src1 vector cache\n"
                                             "    float sumf[nr]={0.f};\n"
                                             "\n"
                                             "    const int ix = tiisg/2;\n"
                                             "    const int il = 8*(tiisg%2);\n"
                                             "\n"
                                             "    device const float * yb = y + ix * QK4_0 + il;\n"
                                             "\n"
                                             "    // each thread in a SIMD group deals with half a block.\n"
                                             "    for (int ib = ix; ib < nb; ib += nw/2) {\n"
                                             "        float sumy = 0;\n"
                                             "        for (int i = 0; i < 8; i += 2) {\n"
                                             "            sumy += yb[i] + yb[i+1];\n"
                                             "            yl[i+0] = yb[i+ 0];\n"
                                             "            yl[i+1] = yb[i+ 1]/256.f;\n"
                                             "            sumy += yb[i+16] + yb[i+17];\n"
                                             "            yl[i+8] = yb[i+16]/16.f;\n"
                                             "            yl[i+9] = yb[i+17]/4096.f;\n"
                                             "        }\n"
                                             "\n"
                                             "        for (int row = 0; row < nr; row++) {\n"
                                             "            sumf[row] += block_q_n_dot_y(x+ib+row*nb, sumy, yl, il);\n"
                                             "        }\n"
                                             "\n"
                                             "        yb += QK4_0 * 16;\n"
                                             "    }\n"
                                             "\n"
                                             "    for (int row = 0; row < nr; ++row) {\n"
                                             "        const float tot = simd_sum(sumf[row]);\n"
                                             "        if (tiisg == 0 && first_row + row < ne01) {\n"
                                             "            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_mul_mat_q4_0_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "    mul_vec_q_n_f32<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,gqa,tgpig,tiisg,sgitg);\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_mul_mat_q4_1_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "     mul_vec_q_n_f32<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,gqa,tgpig,tiisg,sgitg);\n"
                                             "}\n"
                                             "\n"
                                             "#define NB_Q8_0 8\n"
                                             "\n"
                                             "kernel void kernel_mul_mat_q8_0_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "    const int nr  = N_DST;\n"
                                             "    const int nsg = N_SIMDGROUP;\n"
                                             "    const int nw  = N_SIMDWIDTH;\n"
                                             "\n"
                                             "    const int nb = ne00/QK8_0;\n"
                                             "    const int r0 = tgpig.x;\n"
                                             "    const int r1 = tgpig.y;\n"
                                             "    const int im = tgpig.z;\n"
                                             "    const int first_row = (r0 * nsg + sgitg) * nr;\n"
                                             "    const uint offset0 = first_row * nb + im/gqa*(nb*ne0);\n"
                                             "    device const block_q8_0 * x = (device const block_q8_0 *) src0 + offset0;\n"
                                             "    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;\n"
                                             "\n"
                                             "    float yl[NB_Q8_0];\n"
                                             "    float sumf[nr]={0.f};\n"
                                             "\n"
                                             "    const int ix = tiisg/4;\n"
                                             "    const int il = tiisg%4;\n"
                                             "\n"
                                             "    device const float * yb = y + ix * QK8_0 + NB_Q8_0*il;\n"
                                             "\n"
                                             "    // each thread in a SIMD group deals with NB_Q8_0 quants at a time\n"
                                             "    for (int ib = ix; ib < nb; ib += nw/4) {\n"
                                             "        for (int i = 0; i < NB_Q8_0; ++i) {\n"
                                             "            yl[i] = yb[i];\n"
                                             "        }\n"
                                             "\n"
                                             "        for (int row = 0; row < nr; row++) {\n"
                                             "            device const int8_t * qs = x[ib+row*nb].qs + NB_Q8_0*il;\n"
                                             "            float sumq = 0.f;\n"
                                             "            for (int iq = 0; iq < NB_Q8_0; ++iq) {\n"
                                             "                sumq += qs[iq] * yl[iq];\n"
                                             "            }\n"
                                             "            sumf[row] += sumq*x[ib+row*nb].d;\n"
                                             "        }\n"
                                             "\n"
                                             "        yb += NB_Q8_0 * nw;\n"
                                             "    }\n"
                                             "\n"
                                             "    for (int row = 0; row < nr; ++row) {\n"
                                             "        const float tot = simd_sum(sumf[row]);\n"
                                             "        if (tiisg == 0 && first_row + row < ne01) {\n"
                                             "            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_mul_mat_f16_f32_1row(\n"
                                             "        device const  char * src0,\n"
                                             "        device const  char * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        constant  uint64_t & nb00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb02,\n"
                                             "        constant   int64_t & ne10,\n"
                                             "        constant   int64_t & ne11,\n"
                                             "        constant   int64_t & ne12,\n"
                                             "        constant  uint64_t & nb10,\n"
                                             "        constant  uint64_t & nb11,\n"
                                             "        constant  uint64_t & nb12,\n"
                                             "        constant   int64_t & ne0,\n"
                                             "        constant   int64_t & ne1,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]]) {\n"
                                             "\n"
                                             "    const int64_t r0 = tgpig.x;\n"
                                             "    const int64_t r1 = tgpig.y;\n"
                                             "    const int64_t im = tgpig.z;\n"
                                             "\n"
                                             "    device const half  * x = (device const half  *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);\n"
                                             "    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);\n"
                                             "\n"
                                             "    float sumf = 0;\n"
                                             "    if (ne00 < 128) {\n"
                                             "        for (int i = tiisg; i < ne00; i += 32) {\n"
                                             "            sumf += (float) x[i] * (float) y[i];\n"
                                             "        }\n"
                                             "        float all_sum = simd_sum(sumf);\n"
                                             "        if (tiisg == 0) {\n"
                                             "            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;\n"
                                             "        }\n"
                                             "    } else {\n"
                                             "        device const half4  * x4 = (device const half4  *) x;\n"
                                             "        device const float4 * y4 = (device const float4 *) y;\n"
                                             "        for (int i = tiisg; i < ne00/4; i += 32) {\n"
                                             "            for (int k = 0; k < 4; ++k) sumf += (float)x4[i][k] * y4[i][k];\n"
                                             "        }\n"
                                             "        float all_sum = simd_sum(sumf);\n"
                                             "        if (tiisg == 0) {\n"
                                             "            for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) x[i] * y[i];\n"
                                             "            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;\n"
                                             "        }\n"
                                             "    }\n"
                                             "\n"
                                             "}\n"
                                             "\n"
                                             "#define N_F16_F32 4\n"
                                             "\n"
                                             "kernel void kernel_mul_mat_f16_f32(\n"
                                             "        device const  char * src0,\n"
                                             "        device const  char * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        constant  uint64_t & nb00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb02,\n"
                                             "        constant   int64_t & ne10,\n"
                                             "        constant   int64_t & ne11,\n"
                                             "        constant   int64_t & ne12,\n"
                                             "        constant  uint64_t & nb10,\n"
                                             "        constant  uint64_t & nb11,\n"
                                             "        constant  uint64_t & nb12,\n"
                                             "        constant   int64_t & ne0,\n"
                                             "        constant   int64_t & ne1,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]]) {\n"
                                             "\n"
                                             "    const int64_t r0 = tgpig.x;\n"
                                             "    const int64_t rb = tgpig.y*N_F16_F32;\n"
                                             "    const int64_t im = tgpig.z;\n"
                                             "\n"
                                             "    device const half * x = (device const half *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);\n"
                                             "\n"
                                             "    if (ne00 < 128) {\n"
                                             "        for (int row = 0; row < N_F16_F32; ++row) {\n"
                                             "            int r1 = rb + row;\n"
                                             "            if (r1 >= ne11) {\n"
                                             "                break;\n"
                                             "            }\n"
                                             "\n"
                                             "            device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);\n"
                                             "\n"
                                             "            float sumf = 0;\n"
                                             "            for (int i = tiisg; i < ne00; i += 32) {\n"
                                             "                sumf += (float) x[i] * (float) y[i];\n"
                                             "            }\n"
                                             "\n"
                                             "            float all_sum = simd_sum(sumf);\n"
                                             "            if (tiisg == 0) {\n"
                                             "                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;\n"
                                             "            }\n"
                                             "        }\n"
                                             "    } else {\n"
                                             "        device const half4 * x4 = (device const half4 *)x;\n"
                                             "        for (int row = 0; row < N_F16_F32; ++row) {\n"
                                             "            int r1 = rb + row;\n"
                                             "            if (r1 >= ne11) {\n"
                                             "                break;\n"
                                             "            }\n"
                                             "\n"
                                             "            device const float  * y  = (device const float  *) (src1 + r1*nb11 + im*nb12);\n"
                                             "            device const float4 * y4 = (device const float4 *) y;\n"
                                             "\n"
                                             "            float sumf = 0;\n"
                                             "            for (int i = tiisg; i < ne00/4; i += 32) {\n"
                                             "                for (int k = 0; k < 4; ++k) sumf += (float) x4[i][k] * y4[i][k];\n"
                                             "            }\n"
                                             "\n"
                                             "            float all_sum = simd_sum(sumf);\n"
                                             "            if (tiisg == 0) {\n"
                                             "                for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) x[i] * y[i];\n"
                                             "                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;\n"
                                             "            }\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "// Assumes row size (ne00) is a multiple of 4\n"
                                             "kernel void kernel_mul_mat_f16_f32_l4(\n"
                                             "        device const  char * src0,\n"
                                             "        device const  char * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        constant  uint64_t & nb00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb02,\n"
                                             "        constant   int64_t & ne10,\n"
                                             "        constant   int64_t & ne11,\n"
                                             "        constant   int64_t & ne12,\n"
                                             "        constant  uint64_t & nb10,\n"
                                             "        constant  uint64_t & nb11,\n"
                                             "        constant  uint64_t & nb12,\n"
                                             "        constant   int64_t & ne0,\n"
                                             "        constant   int64_t & ne1,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]]) {\n"
                                             "\n"
                                             "    const int nrows = ne11;\n"
                                             "    const int64_t r0 = tgpig.x;\n"
                                             "    const int64_t im = tgpig.z;\n"
                                             "\n"
                                             "    device const half4 * x4 = (device const half4 *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);\n"
                                             "\n"
                                             "    for (int r1 = 0; r1 < nrows; ++r1) {\n"
                                             "        device const float4 * y4 = (device const float4 *) (src1 + r1*nb11 + im*nb12);\n"
                                             "\n"
                                             "        float sumf = 0;\n"
                                             "        for (int i = tiisg; i < ne00/4; i += 32) {\n"
                                             "            for (int k = 0; k < 4; ++k) sumf += (float) x4[i][k] * y4[i][k];\n"
                                             "        }\n"
                                             "\n"
                                             "        float all_sum = simd_sum(sumf);\n"
                                             "        if (tiisg == 0) {\n"
                                             "            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_alibi_f32(\n"
                                             "        device const float * src0,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        constant   int64_t & ne03,\n"
                                             "        constant  uint64_t & nb00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb02,\n"
                                             "        constant  uint64_t & nb03,\n"
                                             "        constant   int64_t & ne0,\n"
                                             "        constant   int64_t & ne1,\n"
                                             "        constant   int64_t & ne2,\n"
                                             "        constant   int64_t & ne3,\n"
                                             "        constant  uint64_t & nb0,\n"
                                             "        constant  uint64_t & nb1,\n"
                                             "        constant  uint64_t & nb2,\n"
                                             "        constant  uint64_t & nb3,\n"
                                             "        constant      float & m0,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint3 tpitg[[thread_position_in_threadgroup]],\n"
                                             "        uint3   ntg[[threads_per_threadgroup]]) {\n"
                                             "    const int64_t i03 = tgpig[2];\n"
                                             "    const int64_t i02 = tgpig[1];\n"
                                             "    const int64_t i01 = tgpig[0];\n"
                                             "\n"
                                             "    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n"
                                             "\n"
                                             "    const int64_t i3 = n / (ne2*ne1*ne0);\n"
                                             "    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);\n"
                                             "    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;\n"
                                             "    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);\n"
                                             "\n"
                                             "    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);\n"
                                             "    float m_k = pow(m0, i2 + 1);\n"
                                             "    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {\n"
                                             "        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);\n"
                                             "        dst_data[i00] = src[0] + m_k * (i00 - ne00 + 1);\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_rope(\n"
                                             "        device const  void * src0,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        constant   int64_t & ne03,\n"
                                             "        constant  uint64_t & nb00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb02,\n"
                                             "        constant  uint64_t & nb03,\n"
                                             "        constant   int64_t & ne0,\n"
                                             "        constant   int64_t & ne1,\n"
                                             "        constant   int64_t & ne2,\n"
                                             "        constant   int64_t & ne3,\n"
                                             "        constant  uint64_t & nb0,\n"
                                             "        constant  uint64_t & nb1,\n"
                                             "        constant  uint64_t & nb2,\n"
                                             "        constant  uint64_t & nb3,\n"
                                             "        constant       int & n_past,\n"
                                             "        constant       int & n_dims,\n"
                                             "        constant       int & mode,\n"
                                             "        constant     float & freq_base,\n"
                                             "        constant     float & freq_scale,\n"
                                             "        uint  tiitg[[thread_index_in_threadgroup]],\n"
                                             "        uint3 tptg[[threads_per_threadgroup]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]]) {\n"
                                             "    const int64_t i3 = tgpig[2];\n"
                                             "    const int64_t i2 = tgpig[1];\n"
                                             "    const int64_t i1 = tgpig[0];\n"
                                             "\n"
                                             "    const bool is_neox = mode & 2;\n"
                                             "\n"
                                             "    const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);\n"
                                             "\n"
                                             "    const float theta_0 = freq_scale * (float)p;\n"
                                             "    const float inv_ndims = -1.f/n_dims;\n"
                                             "\n"
                                             "    if (!is_neox) {\n"
                                             "        for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {\n"
                                             "\n"
                                             "            const float theta = theta_0 * pow(freq_base, inv_ndims*i0);\n"
                                             "            const float cos_theta = cos(theta);\n"
                                             "            const float sin_theta = sin(theta);\n"
                                             "\n"
                                             "            device const float * const src = (device float *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);\n"
                                             "            device       float * dst_data  = (device float *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);\n"
                                             "\n"
                                             "            const float x0 = src[0];\n"
                                             "            const float x1 = src[1];\n"
                                             "\n"
                                             "            dst_data[0] = x0*cos_theta - x1*sin_theta;\n"
                                             "            dst_data[1] = x0*sin_theta + x1*cos_theta;\n"
                                             "        }\n"
                                             "    } else {\n"
                                             "        for (int64_t ib = 0; ib < ne0/n_dims; ++ib) {\n"
                                             "            for (int64_t ic = 2*tiitg; ic < n_dims; ic += 2*tptg.x) {\n"
                                             "\n"
                                             "                const float theta = theta_0 * pow(freq_base, inv_ndims*ic - ib);\n"
                                             "                const float cos_theta = cos(theta);\n"
                                             "                const float sin_theta = sin(theta);\n"
                                             "\n"
                                             "                const int64_t i0 = ib*n_dims + ic/2;\n"
                                             "\n"
                                             "                device const float * const src = (device float *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);\n"
                                             "                device       float * dst_data  = (device float *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);\n"
                                             "\n"
                                             "                const float x0 = src[0];\n"
                                             "                const float x1 = src[n_dims/2];\n"
                                             "\n"
                                             "                dst_data[0]        = x0*cos_theta - x1*sin_theta;\n"
                                             "                dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;\n"
                                             "            }\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_cpy_f16_f16(\n"
                                             "        device const half * src0,\n"
                                             "        device       half * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        constant   int64_t & ne03,\n"
                                             "        constant  uint64_t & nb00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb02,\n"
                                             "        constant  uint64_t & nb03,\n"
                                             "        constant   int64_t & ne0,\n"
                                             "        constant   int64_t & ne1,\n"
                                             "        constant   int64_t & ne2,\n"
                                             "        constant   int64_t & ne3,\n"
                                             "        constant  uint64_t & nb0,\n"
                                             "        constant  uint64_t & nb1,\n"
                                             "        constant  uint64_t & nb2,\n"
                                             "        constant  uint64_t & nb3,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint3 tpitg[[thread_position_in_threadgroup]],\n"
                                             "        uint3   ntg[[threads_per_threadgroup]]) {\n"
                                             "    const int64_t i03 = tgpig[2];\n"
                                             "    const int64_t i02 = tgpig[1];\n"
                                             "    const int64_t i01 = tgpig[0];\n"
                                             "\n"
                                             "    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n"
                                             "\n"
                                             "    const int64_t i3 = n / (ne2*ne1*ne0);\n"
                                             "    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);\n"
                                             "    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;\n"
                                             "    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);\n"
                                             "\n"
                                             "    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);\n"
                                             "\n"
                                             "    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {\n"
                                             "        device const half * src = (device half *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);\n"
                                             "        dst_data[i00] = src[0];\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_cpy_f32_f16(\n"
                                             "        device const float * src0,\n"
                                             "        device        half * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        constant   int64_t & ne03,\n"
                                             "        constant  uint64_t & nb00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb02,\n"
                                             "        constant  uint64_t & nb03,\n"
                                             "        constant   int64_t & ne0,\n"
                                             "        constant   int64_t & ne1,\n"
                                             "        constant   int64_t & ne2,\n"
                                             "        constant   int64_t & ne3,\n"
                                             "        constant  uint64_t & nb0,\n"
                                             "        constant  uint64_t & nb1,\n"
                                             "        constant  uint64_t & nb2,\n"
                                             "        constant  uint64_t & nb3,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint3 tpitg[[thread_position_in_threadgroup]],\n"
                                             "        uint3   ntg[[threads_per_threadgroup]]) {\n"
                                             "    const int64_t i03 = tgpig[2];\n"
                                             "    const int64_t i02 = tgpig[1];\n"
                                             "    const int64_t i01 = tgpig[0];\n"
                                             "\n"
                                             "    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n"
                                             "\n"
                                             "    const int64_t i3 = n / (ne2*ne1*ne0);\n"
                                             "    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);\n"
                                             "    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;\n"
                                             "    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);\n"
                                             "\n"
                                             "    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);\n"
                                             "\n"
                                             "    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {\n"
                                             "        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);\n"
                                             "\n"
                                             "        dst_data[i00] = src[0];\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_cpy_f32_f32(\n"
                                             "        device const float * src0,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01,\n"
                                             "        constant   int64_t & ne02,\n"
                                             "        constant   int64_t & ne03,\n"
                                             "        constant  uint64_t & nb00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb02,\n"
                                             "        constant  uint64_t & nb03,\n"
                                             "        constant   int64_t & ne0,\n"
                                             "        constant   int64_t & ne1,\n"
                                             "        constant   int64_t & ne2,\n"
                                             "        constant   int64_t & ne3,\n"
                                             "        constant  uint64_t & nb0,\n"
                                             "        constant  uint64_t & nb1,\n"
                                             "        constant  uint64_t & nb2,\n"
                                             "        constant  uint64_t & nb3,\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint3 tpitg[[thread_position_in_threadgroup]],\n"
                                             "        uint3   ntg[[threads_per_threadgroup]]) {\n"
                                             "    const int64_t i03 = tgpig[2];\n"
                                             "    const int64_t i02 = tgpig[1];\n"
                                             "    const int64_t i01 = tgpig[0];\n"
                                             "\n"
                                             "    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n"
                                             "\n"
                                             "    const int64_t i3 = n / (ne2*ne1*ne0);\n"
                                             "    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);\n"
                                             "    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;\n"
                                             "    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);\n"
                                             "\n"
                                             "    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);\n"
                                             "\n"
                                             "    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {\n"
                                             "        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);\n"
                                             "\n"
                                             "        dst_data[i00] = src[0];\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "//============================================ k-quants ======================================================\n"
                                             "\n"
                                             "#ifndef QK_K\n"
                                             "#define QK_K 256\n"
                                             "#else\n"
                                             "static_assert(QK_K == 256 || QK_K == 64, \"QK_K must be 256 or 64\");\n"
                                             "#endif\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "#define K_SCALE_SIZE 12\n"
                                             "#else\n"
                                             "#define K_SCALE_SIZE 4\n"
                                             "#endif\n"
                                             "\n"
                                             "typedef struct {\n"
                                             "    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits\n"
                                             "    uint8_t qs[QK_K/4];      // quants\n"
                                             "    half d;           // super-block scale for quantized scales\n"
                                             "    half dmin;        // super-block scale for quantized mins\n"
                                             "} block_q2_K;\n"
                                             "// 84 bytes / block\n"
                                             "\n"
                                             "typedef struct {\n"
                                             "    uint8_t hmask[QK_K/8];     // quants - high bit\n"
                                             "    uint8_t qs[QK_K/4];        // quants - low 2 bits\n"
                                             "#if QK_K == 64\n"
                                             "    uint8_t scales[2];\n"
                                             "#else\n"
                                             "    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits\n"
                                             "#endif\n"
                                             "    half d;             // super-block scale\n"
                                             "} block_q3_K;\n"
                                             "\n"
                                             "#if QK_K == 64\n"
                                             "typedef struct {\n"
                                             "    half    d[2];          // super-block scales/mins\n"
                                             "    uint8_t scales[2];\n"
                                             "    uint8_t qs[QK_K/2];    // 4-bit quants\n"
                                             "} block_q4_K;\n"
                                             "#else\n"
                                             "typedef struct {\n"
                                             "    half d;             // super-block scale for quantized scales\n"
                                             "    half dmin;          // super-block scale for quantized mins\n"
                                             "    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits\n"
                                             "    uint8_t qs[QK_K/2];        // 4--bit quants\n"
                                             "} block_q4_K;\n"
                                             "#endif\n"
                                             "\n"
                                             "#if QK_K == 64\n"
                                             "typedef struct {\n"
                                             "    half  d;                     // super-block scales/mins\n"
                                             "    int8_t  scales[QK_K/16];     // 8-bit block scales\n"
                                             "    uint8_t qh[QK_K/8];          // quants, high bit\n"
                                             "    uint8_t qs[QK_K/2];          // quants, low 4 bits\n"
                                             "} block_q5_K;\n"
                                             "#else\n"
                                             "typedef struct {\n"
                                             "    half d;                      // super-block scale for quantized scales\n"
                                             "    half dmin;                   // super-block scale for quantized mins\n"
                                             "    uint8_t scales[3*QK_K/64];   // scales and mins, quantized with 6 bits\n"
                                             "    uint8_t qh[QK_K/8];          // quants, high bit\n"
                                             "    uint8_t qs[QK_K/2];          // quants, low 4 bits\n"
                                             "} block_q5_K;\n"
                                             "// 176 bytes / block\n"
                                             "#endif\n"
                                             "\n"
                                             "typedef struct {\n"
                                             "    uint8_t ql[QK_K/2];      // quants, lower 4 bits\n"
                                             "    uint8_t qh[QK_K/4];      // quants, upper 2 bits\n"
                                             "    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits\n"
                                             "    half d;                  // super-block scale\n"
                                             "} block_q6_K;\n"
                                             "// 210 bytes / block\n"
                                             "\n"
                                             "static inline uchar4 get_scale_min_k4(int j, device const uint8_t * q) {\n"
                                             "    uchar4 r;\n"
                                             "    if (j < 4) {\n"
                                             "        r[0] = q[j+0] & 63;\n"
                                             "        r[2] = q[j+1] & 63;\n"
                                             "        r[1] = q[j+4] & 63;\n"
                                             "        r[3] = q[j+5] & 63;\n"
                                             "    } else {\n"
                                             "        r[0] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);\n"
                                             "        r[2] = (q[j+5] & 0xF) | ((q[j-3] >> 6) << 4);\n"
                                             "        r[1] = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);\n"
                                             "        r[3] = (q[j+5] >>  4) | ((q[j+1] >> 6) << 4);\n"
                                             "    }\n"
                                             "    return r;\n"
                                             "}\n"
                                             "\n"
                                             "//====================================== dot products =========================\n"
                                             "\n"
                                             "kernel void kernel_mul_mat_q2_K_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "\n"
                                             "    const int nb = ne00/QK_K;\n"
                                             "    const int r0 = tgpig.x;\n"
                                             "    const int r1 = tgpig.y;\n"
                                             "    const int r2 = tgpig.z;\n"
                                             "\n"
                                             "    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;\n"
                                             "    const int ib_row = first_row * nb;\n"
                                             "    const uint offset0 = r2/gqa*(nb*ne0);\n"
                                             "    device const block_q2_K * x = (device const block_q2_K *) src0 + ib_row + offset0;\n"
                                             "    device const float      * y = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;\n"
                                             "    float yl[32];\n"
                                             "    float sumf[N_DST]={0.f}, all_sum;\n"
                                             "\n"
                                             "    const int step = sizeof(block_q2_K) * nb;\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "    const int ix = tiisg/8;  // 0...3\n"
                                             "    const int it = tiisg%8;  // 0...7\n"
                                             "    const int im = it/4;     // 0 or 1\n"
                                             "    const int ir = it%4;     // 0...3\n"
                                             "    const int is = (8*ir)/16;// 0 or 1\n"
                                             "\n"
                                             "    device const float * y4 = y + ix * QK_K + 128 * im + 8 * ir;\n"
                                             "\n"
                                             "    for (int ib = ix; ib < nb; ib += 4) {\n"
                                             "\n"
                                             "        float4 sumy = {0.f, 0.f, 0.f, 0.f};\n"
                                             "        for (int i = 0; i < 8; ++i) {\n"
                                             "            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];\n"
                                             "            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];\n"
                                             "            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];\n"
                                             "            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];\n"
                                             "        }\n"
                                             "\n"
                                             "        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*im + is;\n"
                                             "        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * im + 4 * ir;\n"
                                             "        device const half     * dh = &x[ib].d;\n"
                                             "\n"
                                             "        for (int row = 0; row < N_DST; row++) {\n"
                                             "\n"
                                             "            float4 acc1 = {0.f, 0.f, 0.f, 0.f};\n"
                                             "            float4 acc2 = {0.f, 0.f, 0.f, 0.f};\n"
                                             "            for (int i = 0; i < 8; i += 2) {\n"
                                             "                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);\n"
                                             "                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);\n"
                                             "                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);\n"
                                             "                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);\n"
                                             "                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);\n"
                                             "                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);\n"
                                             "                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);\n"
                                             "                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);\n"
                                             "            }\n"
                                             "            float dall = dh[0];\n"
                                             "            float dmin = dh[1] * 1.f/16.f;\n"
                                             "            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +\n"
                                             "                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +\n"
                                             "                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +\n"
                                             "                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -\n"
                                             "                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));\n"
                                             "\n"
                                             "            qs += step/2;\n"
                                             "            sc += step;\n"
                                             "            dh += step/2;\n"
                                             "        }\n"
                                             "\n"
                                             "        y4 += 4 * QK_K;\n"
                                             "    }\n"
                                             "#else\n"
                                             "    const int ix = tiisg/2;  // 0...15\n"
                                             "    const int it = tiisg%2;  // 0...1\n"
                                             "\n"
                                             "    device const float * y4 = y + ix * QK_K + 8 * it;\n"
                                             "\n"
                                             "    for (int ib = ix; ib < nb; ib += 16) {\n"
                                             "\n"
                                             "        float4 sumy = {0.f, 0.f, 0.f, 0.f};\n"
                                             "        for (int i = 0; i < 8; ++i) {\n"
                                             "            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];\n"
                                             "            yl[i+ 8] = y4[i+16]; sumy[1] += yl[i+ 8];\n"
                                             "            yl[i+16] = y4[i+32]; sumy[2] += yl[i+16];\n"
                                             "            yl[i+24] = y4[i+48]; sumy[3] += yl[i+24];\n"
                                             "        }\n"
                                             "\n"
                                             "        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales;\n"
                                             "        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 4 * it;\n"
                                             "        device const half     * dh = &x[ib].d;\n"
                                             "\n"
                                             "        for (int row = 0; row < N_DST; row++) {\n"
                                             "\n"
                                             "            float4 acc1 = {0.f, 0.f, 0.f, 0.f};\n"
                                             "            float4 acc2 = {0.f, 0.f, 0.f, 0.f};\n"
                                             "            for (int i = 0; i < 8; i += 2) {\n"
                                             "                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);\n"
                                             "                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);\n"
                                             "                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);\n"
                                             "                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);\n"
                                             "                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);\n"
                                             "                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);\n"
                                             "                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);\n"
                                             "                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);\n"
                                             "            }\n"
                                             "\n"
                                             "            float dall = dh[0];\n"
                                             "            float dmin = dh[1];\n"
                                             "            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +\n"
                                             "                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[1] & 0xF) * 1.f/ 4.f +\n"
                                             "                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[2] & 0xF) * 1.f/16.f +\n"
                                             "                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[3] & 0xF) * 1.f/64.f) -\n"
                                             "                         dmin * (sumy[0] * (sc[0] >> 4) + sumy[1] * (sc[1] >> 4) + sumy[2] * (sc[2] >> 4) + sumy[3] * (sc[3] >> 4));\n"
                                             "\n"
                                             "            qs += step/2;\n"
                                             "            sc += step;\n"
                                             "            dh += step/2;\n"
                                             "        }\n"
                                             "\n"
                                             "        y4 += 16 * QK_K;\n"
                                             "    }\n"
                                             "#endif\n"
                                             "\n"
                                             "    for (int row = 0; row < N_DST; ++row) {\n"
                                             "        all_sum = simd_sum(sumf[row]);\n"
                                             "        if (tiisg == 0) {\n"
                                             "            dst[r1*ne0 + r2*ne0*ne1 + first_row + row] = all_sum;\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "kernel void kernel_mul_mat_q3_K_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "\n"
                                             "    const int nb = ne00/QK_K;\n"
                                             "\n"
                                             "    const int64_t r0 = tgpig.x;\n"
                                             "    const int64_t r1 = tgpig.y;\n"
                                             "    const int64_t r2 = tgpig.z;\n"
                                             "\n"
                                             "    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;\n"
                                             "    const uint offset0 = r2/gqa*(nb*ne0);\n"
                                             "    device const block_q3_K * x = (device const block_q3_K *) src0 + first_row*nb + offset0;\n"
                                             "    device const float     * yy = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;\n"
                                             "\n"
                                             "    float yl[32];\n"
                                             "\n"
                                             "    const uint16_t kmask1 = 0x3030;\n"
                                             "    const uint16_t kmask2 = 0x0f0f;\n"
                                             "\n"
                                             "    const int tid = tiisg/4;\n"
                                             "    const int ix  = tiisg%4;\n"
                                             "    const int ip  = tid/4;          // 0 or 1\n"
                                             "    const int il  = 2*((tid%4)/2);  // 0 or 2\n"
                                             "    const int ir  = tid%2;\n"
                                             "    const int n   = 8;\n"
                                             "    const int l0  = n*ir;\n"
                                             "\n"
                                             "    // One would think that the Metal compiler would figure out that ip and il can only have\n"
                                             "    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it\n"
                                             "    // with these two tales.\n"
                                             "    //\n"
                                             "    // Possible masks for the high bit\n"
                                             "    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0\n"
                                             "                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2\n"
                                             "                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0\n"
                                             "                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2\n"
                                             "\n"
                                             "    // Possible masks for the low 2 bits\n"
                                             "    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};\n"
                                             "\n"
                                             "    const ushort4 hm = mm[2*ip + il/2];\n"
                                             "\n"
                                             "    const int shift = 2*il;\n"
                                             "    const float    v1 = il == 0 ? 4.f : 64.f;\n"
                                             "    const float    v2 = 4.f * v1;\n"
                                             "\n"
                                             "    const uint16_t s_shift1 = 4*ip;\n"
                                             "    const uint16_t s_shift2 = s_shift1 + il;\n"
                                             "\n"
                                             "    const int q_offset = 32*ip + l0;\n"
                                             "    const int y_offset = 128*ip + 32*il + l0;\n"
                                             "\n"
                                             "    const int step = sizeof(block_q3_K) * nb / 2;\n"
                                             "\n"
                                             "    device const float * y1 = yy + ix*QK_K + y_offset;\n"
                                             "\n"
                                             "    uint32_t scales32, aux32;\n"
                                             "    thread uint16_t * scales16 = (thread uint16_t *)&scales32;\n"
                                             "    thread const int8_t * scales = (thread const int8_t *)&scales32;\n"
                                             "\n"
                                             "    float sumf1[2] = {0.f};\n"
                                             "    float sumf2[2] = {0.f};\n"
                                             "    for (int i = ix; i < nb; i += 4) {\n"
                                             "\n"
                                             "        for (int l = 0; l < 8; ++l) {\n"
                                             "            yl[l+ 0] = y1[l+ 0];\n"
                                             "            yl[l+ 8] = y1[l+16];\n"
                                             "            yl[l+16] = y1[l+32];\n"
                                             "            yl[l+24] = y1[l+48];\n"
                                             "        }\n"
                                             "\n"
                                             "        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);\n"
                                             "        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);\n"
                                             "        device const uint16_t * a = (device const uint16_t *)(x[i].scales);\n"
                                             "        device const half * dh = &x[i].d;\n"
                                             "\n"
                                             "        for (int row = 0; row < 2; ++row) {\n"
                                             "\n"
                                             "            const float d_all = (float)dh[0];\n"
                                             "\n"
                                             "            scales16[0] = a[4];\n"
                                             "            scales16[1] = a[5];\n"
                                             "            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;\n"
                                             "            scales16[0] = a[il+0];\n"
                                             "            scales16[1] = a[il+1];\n"
                                             "            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;\n"
                                             "\n"
                                             "            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;\n"
                                             "            for (int l = 0; l < n; l += 2) {\n"
                                             "                const int32_t qs = q[l/2];\n"
                                             "                s1 += yl[l+0] * (qs & qm[il/2][0]);\n"
                                             "                s2 += yl[l+1] * (qs & qm[il/2][1]);\n"
                                             "                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);\n"
                                             "                s4 += yl[l+16] * (qs & qm[il/2][2]);\n"
                                             "                s5 += yl[l+17] * (qs & qm[il/2][3]);\n"
                                             "                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);\n"
                                             "            }\n"
                                             "            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);\n"
                                             "            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);\n"
                                             "            sumf1[row] += d1 * (scales[0] - 32);\n"
                                             "            sumf2[row] += d2 * (scales[2] - 32);\n"
                                             "\n"
                                             "            s1 = s2 = s3 = s4 = s5 = s6 = 0;\n"
                                             "            for (int l = 0; l < n; l += 2) {\n"
                                             "                const int32_t qs = q[l/2+8];\n"
                                             "                s1 += yl[l+8] * (qs & qm[il/2][0]);\n"
                                             "                s2 += yl[l+9] * (qs & qm[il/2][1]);\n"
                                             "                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);\n"
                                             "                s4 += yl[l+24] * (qs & qm[il/2][2]);\n"
                                             "                s5 += yl[l+25] * (qs & qm[il/2][3]);\n"
                                             "                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);\n"
                                             "            }\n"
                                             "            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);\n"
                                             "            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);\n"
                                             "            sumf1[row] += d1 * (scales[1] - 32);\n"
                                             "            sumf2[row] += d2 * (scales[3] - 32);\n"
                                             "\n"
                                             "            q  += step;\n"
                                             "            h  += step;\n"
                                             "            a  += step;\n"
                                             "            dh += step;\n"
                                             "\n"
                                             "        }\n"
                                             "\n"
                                             "        y1 += 4 * QK_K;\n"
                                             "\n"
                                             "    }\n"
                                             "\n"
                                             "    for (int row = 0; row < 2; ++row) {\n"
                                             "        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);\n"
                                             "        sumf1[row] = simd_sum(sumf);\n"
                                             "    }\n"
                                             "    if (tiisg == 0) {\n"
                                             "        for (int row = 0; row < 2; ++row) {\n"
                                             "            dst[r1*ne0 + r2*ne0*ne1 + first_row + row] = sumf1[row];\n"
                                             "        }\n"
                                             "    }\n"
                                             "\n"
                                             "}\n"
                                             "#else\n"
                                             "kernel void kernel_mul_mat_q3_K_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "\n"
                                             "    const int nb = ne00/QK_K;\n"
                                             "\n"
                                             "    const int64_t r0 = tgpig.x;\n"
                                             "    const int64_t r1 = tgpig.y;\n"
                                             "    const int64_t r2 = tgpig.z;\n"
                                             "\n"
                                             "    const int row = 2 * r0 + sgitg;\n"
                                             "    const uint offset0 = r2/gqa*(nb*ne0);\n"
                                             "    device const block_q3_K * x = (device const block_q3_K *) src0 + row*nb + offset0;\n"
                                             "    device const float     * yy = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;\n"
                                             "    const int ix = tiisg/4;\n"
                                             "    const int il = 4 * (tiisg%4);// 0, 4, 8, 12\n"
                                             "    const int im = il/8;         // 0, 0, 1, 1\n"
                                             "    const int in = il%8;         // 0, 4, 0, 4\n"
                                             "\n"
                                             "    float2 sum = {0.f, 0.f};\n"
                                             "\n"
                                             "    for (int i = ix; i < nb; i += 8) {\n"
                                             "\n"
                                             "        const float d_all = (float)(x[i].d);\n"
                                             "\n"
                                             "        device const uint16_t * q = (device const uint16_t *)(x[i].qs + il);\n"
                                             "        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + in);\n"
                                             "        device const uint16_t * s = (device const uint16_t *)(x[i].scales);\n"
                                             "        device const float    * y = yy + i * QK_K + il;\n"
                                             "\n"
                                             "        const float d1 = d_all * ((int32_t)(s[0] & 0x000F) - 8);\n"
                                             "        const float d2 = d_all * ((int32_t)(s[0] & 0x00F0) - 128) * 1.f/64.f;\n"
                                             "        const float d3 = d_all * ((int32_t)(s[0] & 0x0F00) - 2048) * 1.f/4096.f;\n"
                                             "        const float d4 = d_all * ((int32_t)(s[0] & 0xF000) - 32768) * 1.f/262144.f;\n"
                                             "\n"
                                             "        for (int l = 0; l < 4; l += 2) {\n"
                                             "            const uint16_t hm = h[l/2] >> im;\n"
                                             "            sum[0] += y[l+ 0] * d1 * ((int32_t)(q[l/2] & 0x0003) - ((hm & 0x0001) ? 0 :  4))\n"
                                             "                    + y[l+16] * d2 * ((int32_t)(q[l/2] & 0x000c) - ((hm & 0x0004) ? 0 : 16))\n"
                                             "                    + y[l+32] * d3 * ((int32_t)(q[l/2] & 0x0030) - ((hm & 0x0010) ? 0 : 64))\n"
                                             "                    + y[l+48] * d4 * ((int32_t)(q[l/2] & 0x00c0) - ((hm & 0x0040) ? 0 : 256));\n"
                                             "            sum[1] += y[l+ 1] * d1 * ((int32_t)(q[l/2] & 0x0300) - ((hm & 0x0100) ? 0 : 1024))\n"
                                             "                    + y[l+17] * d2 * ((int32_t)(q[l/2] & 0x0c00) - ((hm & 0x0400) ? 0 : 4096))\n"
                                             "                    + y[l+33] * d3 * ((int32_t)(q[l/2] & 0x3000) - ((hm & 0x1000) ? 0 : 16384))\n"
                                             "                    + y[l+49] * d4 * ((int32_t)(q[l/2] & 0xc000) - ((hm & 0x4000) ? 0 : 65536));\n"
                                             "        }\n"
                                             "\n"
                                             "    }\n"
                                             "    const float sumf = sum[0] + sum[1] * 1.f/256.f;\n"
                                             "\n"
                                             "    const float tot = simd_sum(sumf);\n"
                                             "    if (tiisg == 0) {\n"
                                             "        dst[r1*ne0 + r2*ne0*ne1 + row] = tot;\n"
                                             "    }\n"
                                             "\n"
                                             "}\n"
                                             "#endif\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "kernel void kernel_mul_mat_q4_K_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "\n"
                                             "    const uint16_t kmask1 = 0x3f3f;\n"
                                             "    const uint16_t kmask2 = 0x0f0f;\n"
                                             "    const uint16_t kmask3 = 0xc0c0;\n"
                                             "\n"
                                             "    const int ix = tiisg/8;  // 0...3\n"
                                             "    const int it = tiisg%8;  // 0...7\n"
                                             "    const int im = it/4;     // 0 or 1\n"
                                             "    const int ir = it%4;     // 0...3\n"
                                             "\n"
                                             "    const int nb = ne00/QK_K;\n"
                                             "    const int r0 = tgpig.x;\n"
                                             "    const int r1 = tgpig.y;\n"
                                             "    const int r2 = tgpig.z;\n"
                                             "    //const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;\n"
                                             "    const int first_row = r0 * N_DST;\n"
                                             "    const int ib_row = first_row * nb;\n"
                                             "    const uint offset0 = r2/gqa*(nb*ne0);\n"
                                             "    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row + offset0;\n"
                                             "    device const float      * y = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;\n"
                                             "    float yl[16];\n"
                                             "    float yh[16];\n"
                                             "    float sumf[N_DST]={0.f}, all_sum;\n"
                                             "\n"
                                             "    const int step = sizeof(block_q4_K) * nb / 2;\n"
                                             "\n"
                                             "    device const float * y4 = y + ix * QK_K + 64 * im + 8 * ir;\n"
                                             "\n"
                                             "    uint16_t sc16[4];\n"
                                             "    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;\n"
                                             "\n"
                                             "    for (int ib = ix; ib < nb; ib += 4) {\n"
                                             "\n"
                                             "        float4 sumy = {0.f, 0.f, 0.f, 0.f};\n"
                                             "        for (int i = 0; i < 8; ++i) {\n"
                                             "            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];\n"
                                             "            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];\n"
                                             "            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];\n"
                                             "            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];\n"
                                             "        }\n"
                                             "\n"
                                             "        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + im;\n"
                                             "        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * im + 4 * ir;\n"
                                             "        device const half     * dh = &x[ib].d;\n"
                                             "\n"
                                             "        for (int row = 0; row < N_DST; row++) {\n"
                                             "\n"
                                             "            sc16[0] = sc[0] & kmask1;\n"
                                             "            sc16[1] = sc[2] & kmask1;\n"
                                             "            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);\n"
                                             "            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);\n"
                                             "\n"
                                             "            device const uint16_t * q2 = q1 + 32;\n"
                                             "\n"
                                             "            float4 acc1 = {0.f, 0.f, 0.f, 0.f};\n"
                                             "            float4 acc2 = {0.f, 0.f, 0.f, 0.f};\n"
                                             "            for (int i = 0; i < 8; i += 2) {\n"
                                             "                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);\n"
                                             "                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);\n"
                                             "                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);\n"
                                             "                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);\n"
                                             "                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);\n"
                                             "                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);\n"
                                             "                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);\n"
                                             "                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);\n"
                                             "            }\n"
                                             "\n"
                                             "            float dall = dh[0];\n"
                                             "            float dmin = dh[1];\n"
                                             "            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +\n"
                                             "                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +\n"
                                             "                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +\n"
                                             "                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -\n"
                                             "                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);\n"
                                             "\n"
                                             "            q1 += step;\n"
                                             "            sc += step;\n"
                                             "            dh += step;\n"
                                             "        }\n"
                                             "\n"
                                             "        y4 += 4 * QK_K;\n"
                                             "    }\n"
                                             "\n"
                                             "    for (int row = 0; row < N_DST; ++row) {\n"
                                             "        all_sum = simd_sum(sumf[row]);\n"
                                             "        if (tiisg == 0) {\n"
                                             "            dst[r1*ne0 + r2*ne0*ne1 + first_row + row] = all_sum;\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "#else\n"
                                             "kernel void kernel_mul_mat_q4_K_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "\n"
                                             "    const int ix = tiisg/4;  // 0...7\n"
                                             "    const int it = tiisg%4;  // 0...3\n"
                                             "\n"
                                             "    const int nb = ne00/QK_K;\n"
                                             "    const int r0 = tgpig.x;\n"
                                             "    const int r1 = tgpig.y;\n"
                                             "    const int r2 = tgpig.z;\n"
                                             "    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;\n"
                                             "    const int ib_row = first_row * nb;\n"
                                             "    const uint offset0 = r2/gqa*(nb*ne0);\n"
                                             "    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row + offset0;\n"
                                             "    device const float      * y = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;\n"
                                             "    float yl[8];\n"
                                             "    float yh[8];\n"
                                             "    float sumf[N_DST]={0.f}, all_sum;\n"
                                             "\n"
                                             "    const int step = sizeof(block_q4_K) * nb / 2;\n"
                                             "\n"
                                             "    device const float * y4 = y + ix * QK_K + 8 * it;\n"
                                             "\n"
                                             "    uint16_t sc16[4];\n"
                                             "\n"
                                             "    for (int ib = ix; ib < nb; ib += 8) {\n"
                                             "\n"
                                             "        float2 sumy = {0.f, 0.f};\n"
                                             "        for (int i = 0; i < 8; ++i) {\n"
                                             "            yl[i] = y4[i+ 0]; sumy[0] += yl[i];\n"
                                             "            yh[i] = y4[i+32]; sumy[1] += yh[i];\n"
                                             "        }\n"
                                             "\n"
                                             "        device const uint16_t * sc = (device const uint16_t *)x[ib].scales;\n"
                                             "        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 4 * it;\n"
                                             "        device const half     * dh = x[ib].d;\n"
                                             "\n"
                                             "        for (int row = 0; row < N_DST; row++) {\n"
                                             "\n"
                                             "            sc16[0] = sc[0] & 0x000f;\n"
                                             "            sc16[1] = sc[0] & 0x0f00;\n"
                                             "            sc16[2] = sc[0] & 0x00f0;\n"
                                             "            sc16[3] = sc[0] & 0xf000;\n"
                                             "\n"
                                             "            float2 acc1 = {0.f, 0.f};\n"
                                             "            float2 acc2 = {0.f, 0.f};\n"
                                             "            for (int i = 0; i < 8; i += 2) {\n"
                                             "                acc1[0] += yl[i+0] * (qs[i/2] & 0x000F);\n"
                                             "                acc1[1] += yl[i+1] * (qs[i/2] & 0x0F00);\n"
                                             "                acc2[0] += yh[i+0] * (qs[i/2] & 0x00F0);\n"
                                             "                acc2[1] += yh[i+1] * (qs[i/2] & 0xF000);\n"
                                             "            }\n"
                                             "\n"
                                             "            float dall = dh[0];\n"
                                             "            float dmin = dh[1];\n"
                                             "            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc16[0] +\n"
                                             "                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc16[1] * 1.f/4096.f) -\n"
                                             "                         dmin * 1.f/16.f * (sumy[0] * sc16[2] + sumy[1] * sc16[3] * 1.f/256.f);\n"
                                             "\n"
                                             "            qs += step;\n"
                                             "            sc += step;\n"
                                             "            dh += step;\n"
                                             "        }\n"
                                             "\n"
                                             "        y4 += 8 * QK_K;\n"
                                             "    }\n"
                                             "\n"
                                             "    for (int row = 0; row < N_DST; ++row) {\n"
                                             "        all_sum = simd_sum(sumf[row]);\n"
                                             "        if (tiisg == 0) {\n"
                                             "            dst[r1*ne0+ r2*ne0*ne1 + first_row + row] = all_sum;\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "#endif\n"
                                             "\n"
                                             "kernel void kernel_mul_mat_q5_K_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "\n"
                                             "    const int nb = ne00/QK_K;\n"
                                             "\n"
                                             "    const int64_t r0 = tgpig.x;\n"
                                             "    const int64_t r1 = tgpig.y;\n"
                                             "    const int r2 = tgpig.z;\n"
                                             "\n"
                                             "    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;\n"
                                             "    const uint offset0 = r2/gqa*(nb*ne0);\n"
                                             "    device const block_q5_K * x = (device const block_q5_K *) src0 + first_row*nb + offset0;\n"
                                             "    device const float     * yy = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;\n"
                                             "\n"
                                             "    float sumf[2]={0.f};\n"
                                             "\n"
                                             "    const int step = sizeof(block_q5_K) * nb;\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "#\n"
                                             "    float yl[16], yh[16];\n"
                                             "\n"
                                             "    const uint16_t kmask1 = 0x3f3f;\n"
                                             "    const uint16_t kmask2 = 0x0f0f;\n"
                                             "    const uint16_t kmask3 = 0xc0c0;\n"
                                             "\n"
                                             "    const int tid = tiisg/4;\n"
                                             "    const int ix  = tiisg%4;\n"
                                             "    const int im  = tid/4;\n"
                                             "    const int ir  = tid%4;\n"
                                             "    const int n   = 8;\n"
                                             "\n"
                                             "    const int l0 = n*ir;\n"
                                             "    const int q_offset = 32*im + l0;\n"
                                             "    const int y_offset = 64*im + l0;\n"
                                             "\n"
                                             "    const uint8_t hm1 = 1u << (2*im);\n"
                                             "    const uint8_t hm2 = hm1 << 1;\n"
                                             "    const uint8_t hm3 = hm1 << 4;\n"
                                             "    const uint8_t hm4 = hm2 << 4;\n"
                                             "\n"
                                             "    uint16_t sc16[4];\n"
                                             "    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;\n"
                                             "\n"
                                             "    device const float * y1 = yy + ix*QK_K + y_offset;\n"
                                             "\n"
                                             "    for (int i = ix; i < nb; i += 4) {\n"
                                             "\n"
                                             "        device const uint8_t * q1 = x[i].qs + q_offset;\n"
                                             "        device const uint8_t * qh = x[i].qh + l0;\n"
                                             "        device const half * dh = &x[i].d;\n"
                                             "        device const uint16_t * a = (device const uint16_t *)x[i].scales + im;\n"
                                             "\n"
                                             "        device const float * y2 = y1 + 128;\n"
                                             "        float4 sumy = {0.f, 0.f, 0.f, 0.f};\n"
                                             "        for (int l = 0; l < 8; ++l) {\n"
                                             "            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];\n"
                                             "            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];\n"
                                             "            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];\n"
                                             "            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];\n"
                                             "        }\n"
                                             "\n"
                                             "        for (int row = 0; row < 2; ++row) {\n"
                                             "\n"
                                             "            device const uint8_t * q2 = q1 + 64;\n"
                                             "\n"
                                             "            sc16[0] = a[0] & kmask1;\n"
                                             "            sc16[1] = a[2] & kmask1;\n"
                                             "            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);\n"
                                             "            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);\n"
                                             "\n"
                                             "            float4 acc1 = {0.f};\n"
                                             "            float4 acc2 = {0.f};\n"
                                             "            for (int l = 0; l < n; ++l) {\n"
                                             "                uint8_t h = qh[l];\n"
                                             "                acc1[0] += yl[l+0] * (q1[l] & 0x0F);\n"
                                             "                acc1[1] += yl[l+8] * (q1[l] & 0xF0);\n"
                                             "                acc1[2] += yh[l+0] * (q2[l] & 0x0F);\n"
                                             "                acc1[3] += yh[l+8] * (q2[l] & 0xF0);\n"
                                             "                acc2[0] += h & hm1 ? yl[l+0] : 0.f;\n"
                                             "                acc2[1] += h & hm2 ? yl[l+8] : 0.f;\n"
                                             "                acc2[2] += h & hm3 ? yh[l+0] : 0.f;\n"
                                             "                acc2[3] += h & hm4 ? yh[l+8] : 0.f;\n"
                                             "            }\n"
                                             "            const float dall = dh[0];\n"
                                             "            const float dmin = dh[1];\n"
                                             "            sumf[row] += dall * (sc8[0] * (acc1[0] +  16.f*acc2[0]) +\n"
                                             "                                 sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +\n"
                                             "                                 sc8[4] * (acc1[2] +  16.f*acc2[2]) +\n"
                                             "                                 sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -\n"
                                             "                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);\n"
                                             "\n"
                                             "            q1 += step;\n"
                                             "            qh += step;\n"
                                             "            dh += step/2;\n"
                                             "            a  += step/2;\n"
                                             "\n"
                                             "        }\n"
                                             "\n"
                                             "        y1 += 4 * QK_K;\n"
                                             "\n"
                                             "    }\n"
                                             "#else\n"
                                             "    float yl[8], yh[8];\n"
                                             "\n"
                                             "    const int il = 4 * (tiisg/8);  // 0, 4, 8, 12\n"
                                             "    const int ix = tiisg%8;\n"
                                             "    const int im = il/8;         // 0, 0, 1, 1\n"
                                             "    const int in = il%8;         // 0, 4, 0, 4\n"
                                             "\n"
                                             "    device const float * y = yy + ix*QK_K + il;\n"
                                             "\n"
                                             "    for (int i = ix; i < nb; i += 8) {\n"
                                             "\n"
                                             "        for (int l = 0; l < 4; ++l) {\n"
                                             "            yl[l+0] = y[l+ 0];\n"
                                             "            yl[l+4] = y[l+16];\n"
                                             "            yh[l+0] = y[l+32];\n"
                                             "            yh[l+4] = y[l+48];\n"
                                             "        }\n"
                                             "\n"
                                             "        device const half * dh = &x[i].d;\n"
                                             "        device const uint8_t * q = x[i].qs + il;\n"
                                             "        device const uint8_t * h = x[i].qh + in;\n"
                                             "        device const int8_t  * s = x[i].scales;\n"
                                             "\n"
                                             "        for (int row = 0; row < 2; ++row) {\n"
                                             "\n"
                                             "            const float d = dh[0];\n"
                                             "\n"
                                             "            float2 acc = {0.f, 0.f};\n"
                                             "            for (int l = 0; l < 4; ++l) {\n"
                                             "                const uint8_t hl = h[l] >> im;\n"
                                             "                acc[0] += yl[l+0] * s[0] * ((int16_t)(q[l+ 0] & 0x0F) - (hl & 0x01 ? 0 : 16))\n"
                                             "                        + yl[l+4] * s[1] * ((int16_t)(q[l+16] & 0x0F) - (hl & 0x04 ? 0 : 16));\n"
                                             "                acc[1] += yh[l+0] * s[2] * ((int16_t)(q[l+ 0] & 0xF0) - (hl & 0x10 ? 0 : 256))\n"
                                             "                        + yh[l+4] * s[3] * ((int16_t)(q[l+16] & 0xF0) - (hl & 0x40 ? 0 : 256));\n"
                                             "            }\n"
                                             "            sumf[row] += d * (acc[0] + 1.f/16.f * acc[1]);\n"
                                             "\n"
                                             "            q += step;\n"
                                             "            h += step;\n"
                                             "            s += step;\n"
                                             "            dh += step/2;\n"
                                             "\n"
                                             "        }\n"
                                             "\n"
                                             "        y += 8 * QK_K;\n"
                                             "    }\n"
                                             "#endif\n"
                                             "\n"
                                             "    for (int row = 0; row < 2; ++row) {\n"
                                             "        const float tot = simd_sum(sumf[row]);\n"
                                             "        if (tiisg == 0) {\n"
                                             "            dst[r1*ne0 + r2*ne0*ne1 + first_row + row] = tot;\n"
                                             "        }\n"
                                             "    }\n"
                                             "\n"
                                             "}\n"
                                             "\n"
                                             "kernel void kernel_mul_mat_q6_K_f32(\n"
                                             "        device const  void * src0,\n"
                                             "        device const float * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant   int64_t & ne01[[buffer(4)]],\n"
                                             "        constant   int64_t & ne02[[buffer(5)]],\n"
                                             "        constant   int64_t & ne10[[buffer(9)]],\n"
                                             "        constant   int64_t & ne12[[buffer(11)]],\n"
                                             "        constant   int64_t & ne0[[buffer(15)]],\n"
                                             "        constant   int64_t & ne1[[buffer(16)]],\n"
                                             "        constant   uint    & gqa[[buffer(17)]],\n"
                                             "        uint3 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint tiisg[[thread_index_in_simdgroup]],\n"
                                             "        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "\n"
                                             "    const uint8_t kmask1 = 0x03;\n"
                                             "    const uint8_t kmask2 = 0x0C;\n"
                                             "    const uint8_t kmask3 = 0x30;\n"
                                             "    const uint8_t kmask4 = 0xC0;\n"
                                             "\n"
                                             "    const int nb = ne00/QK_K;\n"
                                             "\n"
                                             "    const int64_t r0 = tgpig.x;\n"
                                             "    const int64_t r1 = tgpig.y;\n"
                                             "    const int r2 = tgpig.z;\n"
                                             "\n"
                                             "    const int row = 2 * r0 + sgitg;\n"
                                             "    const uint offset0 = r2/gqa*(nb*ne0);\n"
                                             "    device const block_q6_K * x = (device const block_q6_K *) src0 + row * nb + offset0;\n"
                                             "    device const float     * yy = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;\n"
                                             "\n"
                                             "    float sumf = 0;\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "    const int tid  = tiisg/2;\n"
                                             "    const int ix   = tiisg%2;\n"
                                             "    const int ip   = tid/8;         // 0 or 1\n"
                                             "    const int il   = tid%8;\n"
                                             "    const int n    = 4;\n"
                                             "    const int l0   = n*il;\n"
                                             "    const int is   = 8*ip + l0/16;\n"
                                             "\n"
                                             "    const int y_offset = 128*ip + l0;\n"
                                             "    const int q_offset_l = 64*ip + l0;\n"
                                             "    const int q_offset_h = 32*ip + l0;\n"
                                             "\n"
                                             "    for (int i = ix; i < nb; i += 2) {\n"
                                             "\n"
                                             "        device const uint8_t * q1 = x[i].ql + q_offset_l;\n"
                                             "        device const uint8_t * q2 = q1 + 32;\n"
                                             "        device const uint8_t * qh = x[i].qh + q_offset_h;\n"
                                             "        device const int8_t  * sc = x[i].scales + is;\n"
                                             "\n"
                                             "        device const float * y = yy + i * QK_K + y_offset;\n"
                                             "\n"
                                             "        const float dall = x[i].d;\n"
                                             "\n"
                                             "        float4 sums = {0.f, 0.f, 0.f, 0.f};\n"
                                             "        for (int l = 0; l < n; ++l) {\n"
                                             "            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);\n"
                                             "            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);\n"
                                             "            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);\n"
                                             "            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);\n"
                                             "        }\n"
                                             "\n"
                                             "        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);\n"
                                             "\n"
                                             "    }\n"
                                             "\n"
                                             "#else\n"
                                             "    const int ix  = tiisg/4;\n"
                                             "    const int il  = 4*(tiisg%4);\n"
                                             "\n"
                                             "    for (int i = ix; i < nb; i += 8) {\n"
                                             "        device const float * y = yy + i * QK_K + il;\n"
                                             "        device const uint8_t * ql = x[i].ql + il;\n"
                                             "        device const uint8_t * qh = x[i].qh + il;\n"
                                             "        device const int8_t  * s  = x[i].scales;\n"
                                             "\n"
                                             "        const float d = x[i].d;\n"
                                             "\n"
                                             "        float4 sums = {0.f, 0.f, 0.f, 0.f};\n"
                                             "        for (int l = 0; l < 4; ++l) {\n"
                                             "            sums[0] += y[l+ 0] * ((int8_t)((ql[l+ 0] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);\n"
                                             "            sums[1] += y[l+16] * ((int8_t)((ql[l+16] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);\n"
                                             "            sums[2] += y[l+32] * ((int8_t)((ql[l+ 0] >>  4) | ((qh[l] & kmask3) >> 0)) - 32);\n"
                                             "            sums[3] += y[l+48] * ((int8_t)((ql[l+16] >>  4) | ((qh[l] & kmask4) >> 2)) - 32);\n"
                                             "        }\n"
                                             "        sumf += d * (sums[0] * s[0] + sums[1] * s[1] + sums[2] * s[2] + sums[3] * s[3]);\n"
                                             "    }\n"
                                             "\n"
                                             "#endif\n"
                                             "\n"
                                             "    const float tot = simd_sum(sumf);\n"
                                             "    if (tiisg == 0) {\n"
                                             "        dst[r1*ne0 + r2*ne0*ne1 + row] = tot;\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "//============================= templates and their specializations =============================\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {\n"
                                             "    half4x4 temp = *(((device half4x4 *)src));\n"
                                             "    for (int i = 0; i < 16; i++){\n"
                                             "        reg[i/4][i%4] = temp[i/4][i%4];\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_q4_0(device const block_q4_0 *xb, short il, thread type4x4 & reg) {\n"
                                             "\n"
                                             "    device const uint16_t * qs = ((device const uint16_t *)xb + 1);\n"
                                             "    const float d1 = il ? (xb->d / 16.h) : xb->d;\n"
                                             "    const float d2 = d1 / 256.f;\n"
                                             "    const float md = -8.h * xb->d;\n"
                                             "    const ushort mask0 = il ? 0x00F0 : 0x000F;\n"
                                             "    const ushort mask1 = mask0 << 8;\n"
                                             "\n"
                                             "    for (int i=0;i<8;i++) {\n"
                                             "        reg[i/2][2*(i%2)+0] = d1 * (qs[i] & mask0) + md;\n"
                                             "        reg[i/2][2*(i%2)+1] = d2 * (qs[i] & mask1) + md;\n"
                                             "    }\n"
                                             "\n"
                                             "}\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_q4_1(device const block_q4_1 *xb, short il, thread type4x4 & reg) {\n"
                                             "\n"
                                             "    device const uint16_t * qs = ((device const uint16_t *)xb + 2);\n"
                                             "    const float d1 = il ? (xb->d / 16.h) : xb->d;\n"
                                             "    const float d2 = d1 / 256.f;\n"
                                             "    const float  m = xb->m;\n"
                                             "    const ushort mask0 = il ? 0x00F0 : 0x000F;\n"
                                             "    const ushort mask1 = mask0 << 8;\n"
                                             "\n"
                                             "    for (int i=0;i<8;i++) {\n"
                                             "        reg[i/2][2*(i%2)+0] = ((qs[i] & mask0) * d1) + m;\n"
                                             "        reg[i/2][2*(i%2)+1] = ((qs[i] & mask1) * d2) + m;\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {\n"
                                             "    device const int8_t * qs = ((device const int8_t *)xb->qs);\n"
                                             "    const half d = xb->d;\n"
                                             "\n"
                                             "    for (int i=0;i<16;i++) {\n"
                                             "        reg[i/4][i%4] = (qs[i + 16*il] * d);\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {\n"
                                             "    const half d = xb->d;\n"
                                             "    const half min = xb->dmin;\n"
                                             "    device const uint8_t * q = (device const uint8_t *)xb->qs;\n"
                                             "    half dl, ml;\n"
                                             "    uint8_t sc = xb->scales[il];\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "    q = q + 32*(il/8) + 16*(il&1);\n"
                                             "    il = (il/2)%4;\n"
                                             "#endif\n"
                                             "    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);\n"
                                             "    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);\n"
                                             "    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);\n"
                                             "    for (int i = 0; i < 16; ++i) {\n"
                                             "        reg[i/4][i%4] = dl * (q[i] & mask) - ml;\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {\n"
                                             "    const half d_all = xb->d;\n"
                                             "    device const uint8_t * q = (device const uint8_t *)xb->qs;\n"
                                             "    device const uint8_t * h = (device const uint8_t *)xb->hmask;\n"
                                             "    device const int8_t * scales = (device const int8_t *)xb->scales;\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "    q = q + 32 * (il/8) + 16 * (il&1);\n"
                                             "    h = h + 16 * (il&1);\n"
                                             "    uint8_t m = 1 << (il/2);\n"
                                             "    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \\\n"
                                             "                                 ((il/4)>0 ? 12  : 3);\n"
                                             "    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;\n"
                                             "    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];\n"
                                             "    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)\n"
                                             "                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);\n"
                                             "    half dl = il<8 ? d_all * (dl_int - 32.h) : d_all * (dl_int / 16.h - 32.h);\n"
                                             "    const half ml = 4.h * dl;\n"
                                             "\n"
                                             "    il = (il/2) & 3;\n"
                                             "    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);\n"
                                             "    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);\n"
                                             "    dl *= coef;\n"
                                             "\n"
                                             "    for (int i = 0; i < 16; ++i) {\n"
                                             "        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);\n"
                                             "    }\n"
                                             "\n"
                                             "#else\n"
                                             "    float    kcoef = il&1 ? 1.f/16.f : 1.f;\n"
                                             "    uint16_t kmask = il&1 ? 0xF0     : 0x0F;\n"
                                             "    float    dl = d_all * ((scales[il/2] & kmask) * kcoef - 8);\n"
                                             "    float    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);\n"
                                             "    uint8_t  mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);\n"
                                             "    uint8_t  m = 1<<(il*2);\n"
                                             "    for (int i = 0; i < 16; ++i) {\n"
                                             "        reg[i/4][i%4] = coef * dl * ((q[i] & mask) - ((h[i%8] & (m * (1 + i/8))) ? 0 : 4.f/coef));\n"
                                             "    }\n"
                                             "#endif\n"
                                             "}\n"
                                             "\n"
                                             "static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {\n"
                                             "    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}\n"
                                             "                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};\n"
                                             "}\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {\n"
                                             "    device const uchar * q = xb->qs;\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "    short is = (il/4) * 2;\n"
                                             "    q = q + (il/4) * 32 + 16 * (il&1);\n"
                                             "    il = il & 3;\n"
                                             "    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);\n"
                                             "    const half d   = il < 2 ? xb->d : xb->d / 16.h;\n"
                                             "    const half min = xb->dmin;\n"
                                             "    const half dl = d * sc[0];\n"
                                             "    const half ml = min * sc[1];\n"
                                             "#else\n"
                                             "    q = q + 16 * (il&1);\n"
                                             "    device const uint8_t * s = xb->scales;\n"
                                             "    device const half2 * dh = (device const half2 *)xb->d;\n"
                                             "    const float2 d = (float2)dh[0];\n"
                                             "    const float dl = il<2 ? d[0] * (s[0]&0xF) : d[0] * (s[1]&0xF)/16.h;\n"
                                             "    const float ml = il<2 ? d[1] * (s[0]>>4)  : d[1] * (s[1]>>4);\n"
                                             "#endif\n"
                                             "    const ushort mask = il<2 ? 0x0F : 0xF0;\n"
                                             "    for (int i = 0; i < 16; ++i) {\n"
                                             "        reg[i/4][i%4] = dl * (q[i] & mask) - ml;\n"
                                             "    }\n"
                                             "\n"
                                             "}\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {\n"
                                             "    device const uint8_t * q  = xb->qs;\n"
                                             "    device const uint8_t * qh = xb->qh;\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "    short is = (il/4) * 2;\n"
                                             "    q  = q + 32 * (il/4) + 16 * (il&1);\n"
                                             "    qh = qh + 16 * (il&1);\n"
                                             "    uint8_t ul = 1 << (il/2);\n"
                                             "    il = il & 3;\n"
                                             "    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);\n"
                                             "    const half d = il < 2 ? xb->d : xb->d / 16.h;\n"
                                             "    const half min = xb->dmin;\n"
                                             "    const half dl = d * sc[0];\n"
                                             "    const half ml = min * sc[1];\n"
                                             "\n"
                                             "    const ushort mask = il<2 ? 0x0F : 0xF0;\n"
                                             "    const half qh_val = il<2 ? 16.h : 256.h;\n"
                                             "    for (int i = 0; i < 16; ++i) {\n"
                                             "        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;\n"
                                             "    }\n"
                                             "#else\n"
                                             "    q = q + 16 * (il&1);\n"
                                             "    device const int8_t * s = xb->scales;\n"
                                             "    const float dl = xb->d * s[il];\n"
                                             "    uint8_t m = 1<<(il*2);\n"
                                             "    const float  coef = il<2 ? 1.f  : 1.f/16.f;\n"
                                             "    const ushort mask = il<2 ? 0x0F : 0xF0;\n"
                                             "    for (int i = 0; i < 16; ++i) {\n"
                                             "        reg[i/4][i%4] = coef * dl * ((q[i] & mask) - (qh[i%8] & (m*(1+i/8)) ? 0.f : 16.f/coef));\n"
                                             "    }\n"
                                             "#endif\n"
                                             "}\n"
                                             "\n"
                                             "template <typename type4x4>\n"
                                             "void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {\n"
                                             "    const half d_all = xb->d;\n"
                                             "    device const uint8_t * ql = (device const uint8_t *)xb->ql;\n"
                                             "    device const uint8_t * qh = (device const uint8_t *)xb->qh;\n"
                                             "    device const int8_t * scales = (device const int8_t *)xb->scales;\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "    ql = ql + 64*(il/8) + 32*((il/2)&1) + 16*(il&1);\n"
                                             "    qh = qh + 32*(il/8) + 16*(il&1);\n"
                                             "    half sc = scales[(il%2) + 2 * ((il/2))];\n"
                                             "    il = (il/2) & 3;\n"
                                             "#else\n"
                                             "    ql = ql + 16 * (il&1);\n"
                                             "    half sc = scales[il];\n"
                                             "#endif\n"
                                             "    const uint16_t  kmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);\n"
                                             "    const uint16_t  kmask2 = il>1 ? 0xF0              : 0x0F;\n"
                                             "    const half        coef = il>1 ? 1.f/16.h          : 1.h;\n"
                                             "    const half ml = d_all * sc * 32.h;\n"
                                             "    const half dl = d_all * sc * coef;\n"
                                             "    for (int i = 0; i < 16; ++i) {\n"
                                             "        const half q = il&1 ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))\n"
                                             "                            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));\n"
                                             "        reg[i/4][i%4] = dl * q - ml;\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>\n"
                                             "kernel void kernel_get_rows(\n"
                                             "        device const  void * src0,\n"
                                             "        device const   int * src1,\n"
                                             "        device       float * dst,\n"
                                             "        constant   int64_t & ne00,\n"
                                             "        constant  uint64_t & nb01,\n"
                                             "        constant  uint64_t & nb1,\n"
                                             "        uint                 tgpig[[threadgroup_position_in_grid]],\n"
                                             "        uint                 tiitg[[thread_index_in_threadgroup]],\n"
                                             "        uint                 tptg[[threads_per_threadgroup]]) {\n"
                                             "    const int i = tgpig;\n"
                                             "    const int r = ((device int32_t *) src1)[i];\n"
                                             "\n"
                                             "    for (int ind = tiitg; ind < ne00/16; ind += tptg) {\n"
                                             "        float4x4 temp;\n"
                                             "        dequantize_func(\n"
                                             "            ((device const block_q *) ((device char *) src0 + r*nb01)) + ind/nl, ind%nl, temp);\n"
                                             "        *(((device float4x4 *) ((device char *) dst + i*nb1)) + ind) = temp;\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A\n"
                                             "#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix A\n"
                                             "#define BLOCK_SIZE_K 32\n"
                                             "#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A\n"
                                             "#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B\n"
                                             "#define THREAD_PER_BLOCK 128\n"
                                             "#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers\n"
                                             "#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers\n"
                                             "#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8\n"
                                             "#define SG_MAT_ROW 8\n"
                                             "\n"
                                             "// each block_q contains 16*nl weights\n"
                                             "template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>\n"
                                             "kernel void kernel_mul_mm(device const  uchar * src0,\n"
                                             "                           device const  float * src1,\n"
                                             "                           device        float * dst,\n"
                                             "                           constant    int64_t & ne00,\n"
                                             "                           constant    int64_t & ne02,\n"
                                             "                           constant    int64_t & nb01,\n"
                                             "                           constant    int64_t & nb02,\n"
                                             "                           constant    int64_t & ne12,\n"
                                             "                           constant    int64_t & ne0,\n"
                                             "                           constant    int64_t & ne1,\n"
                                             "                           constant    uint & gqa,\n"
                                             "                           threadgroup   uchar * shared_memory [[threadgroup(0)]],\n"
                                             "                           uint3                 tgpig[[threadgroup_position_in_grid]],\n"
                                             "                           uint                  tiitg[[thread_index_in_threadgroup]],\n"
                                             "                           uint                  sgitg[[simdgroup_index_in_threadgroup]]) {\n"
                                             "\n"
                                             "    threadgroup half * sa = ((threadgroup half *)shared_memory);\n"
                                             "    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);\n"
                                             "\n"
                                             "    const uint r0 = tgpig.y;\n"
                                             "    const uint r1 = tgpig.x;\n"
                                             "    const uint im = tgpig.z;\n"
                                             "    // if this block is of 64x32 shape or smaller\n"
                                             "    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;\n"
                                             "    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;\n"
                                             "    // a thread shouldn't load data outside of the matrix\n"
                                             "    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;\n"
                                             "    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;\n"
                                             "\n"
                                             "    simdgroup_half8x8 ma[4];\n"
                                             "    simdgroup_float8x8 mb[2];\n"
                                             "    simdgroup_float8x8 c_res[8];\n"
                                             "    for (int i = 0; i < 8; i++){\n"
                                             "        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);\n"
                                             "    }\n"
                                             "\n"
                                             "    short il = (tiitg % THREAD_PER_ROW);\n"
                                             "    uint offset0 = im/gqa*nb02; ushort offset1 = il/nl;\n"
                                             "    device const block_q  * x = (device const block_q  *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;\n"
                                             "    device const float * y = src1 + (r1 * BLOCK_SIZE_N + thread_col) * ne00 \\\n"
                                             "                             + BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL) + im * ne00 * ne1;\n"
                                             "\n"
                                             "    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {\n"
                                             "        //load data and store to threadgroup memory\n"
                                             "        half4x4 temp_a;\n"
                                             "        dequantize_func(x, il, temp_a);\n"
                                             "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "        #pragma unroll(16)\n"
                                             "        for (int i = 0; i < 16; i++) {\n"
                                             "            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \\\n"
                                             "            + 16 * (tiitg % THREAD_PER_ROW) + 8 * (i / 8)) \\\n"
                                             "            + (tiitg / THREAD_PER_ROW) % 8 + (i & 7) * 8) = temp_a[i/4][i%4];\n"
                                             "        }\n"
                                             "        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) \\\n"
                                             "                = *((device float2x4 *)y);\n"
                                             "        il = (il + 2 < nl) ? il + 2 : il % 2;\n"
                                             "        x  = (il < 2) ? x + (2+nl-1)/nl : x;\n"
                                             "        y += BLOCK_SIZE_K;\n"
                                             "\n"
                                             "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "        //load matrices from threadgroup memory and conduct outer products\n"
                                             "        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));\n"
                                             "        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));\n"
                                             "        #pragma unroll(4)\n"
                                             "        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {\n"
                                             "            #pragma unroll(4)\n"
                                             "            for (int i = 0; i < 4; i++) {\n"
                                             "                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);\n"
                                             "            }\n"
                                             "            simdgroup_barrier(mem_flags::mem_none);\n"
                                             "            #pragma unroll(2)\n"
                                             "            for (int i = 0; i < 2; i++) {\n"
                                             "                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);\n"
                                             "            }\n"
                                             "\n"
                                             "            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;\n"
                                             "            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;\n"
                                             "            #pragma unroll(8)\n"
                                             "            for (int i = 0; i < 8; i++){\n"
                                             "                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);\n"
                                             "            }\n"
                                             "        }\n"
                                             "    }\n"
                                             "\n"
                                             "    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {\n"
                                             "        device float *C = dst + BLOCK_SIZE_M * r0 + 32 * (sgitg&1) \\\n"
                                             "                          + (BLOCK_SIZE_N * r1 + 16 * (sgitg>>1)) * ne0 + im*ne1*ne0;\n"
                                             "        for (int i = 0; i < 8; i++) {\n"
                                             "            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);\n"
                                             "        }\n"
                                             "    } else {\n"
                                             "        // block is smaller than 64x32, we should avoid writing data outside of the matrix\n"
                                             "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "        threadgroup float *temp_str = ((threadgroup float *)shared_memory) \\\n"
                                             "                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;\n"
                                             "        for (int i = 0; i < 8; i++) {\n"
                                             "            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);\n"
                                             "        }\n"
                                             "\n"
                                             "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                                             "        device float *C = dst + BLOCK_SIZE_M * r0 + (BLOCK_SIZE_N * r1) * ne0 + im*ne1*ne0;\n"
                                             "        if (sgitg==0) {\n"
                                             "            for (int i = 0; i < n_rows; i++) {\n"
                                             "                for (int j = tiitg; j< n_cols; j += BLOCK_SIZE_N) {\n"
                                             "                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);\n"
                                             "                }\n"
                                             "            }\n"
                                             "        }\n"
                                             "    }\n"
                                             "}\n"
                                             "\n"
                                             "#if QK_K == 256\n"
                                             "#define QK_NL 16\n"
                                             "#else\n"
                                             "#define QK_NL 4\n"
                                             "#endif\n"
                                             "\n"
                                             "typedef void (get_rows_t)(device const void *, device const int *, device float *, constant int64_t &, \\\n"
                                             "                          constant uint64_t &, constant uint64_t &, uint, uint, uint);\n"
                                             "\n"
                                             "template [[host_name(\"kernel_get_rows_f16\")]]  kernel get_rows_t kernel_get_rows<half4x4,    1, dequantize_f16>;\n"
                                             "template [[host_name(\"kernel_get_rows_q4_0\")]] kernel get_rows_t kernel_get_rows<block_q4_0, 2, dequantize_q4_0>;\n"
                                             "template [[host_name(\"kernel_get_rows_q4_1\")]] kernel get_rows_t kernel_get_rows<block_q4_1, 2, dequantize_q4_1>;\n"
                                             "template [[host_name(\"kernel_get_rows_q8_0\")]] kernel get_rows_t kernel_get_rows<block_q8_0, 2, dequantize_q8_0>;\n"
                                             "template [[host_name(\"kernel_get_rows_q2_K\")]] kernel get_rows_t kernel_get_rows<block_q2_K, QK_NL, dequantize_q2_K>;\n"
                                             "template [[host_name(\"kernel_get_rows_q3_K\")]] kernel get_rows_t kernel_get_rows<block_q3_K, QK_NL, dequantize_q3_K>;\n"
                                             "template [[host_name(\"kernel_get_rows_q4_K\")]] kernel get_rows_t kernel_get_rows<block_q4_K, QK_NL, dequantize_q4_K>;\n"
                                             "template [[host_name(\"kernel_get_rows_q5_K\")]] kernel get_rows_t kernel_get_rows<block_q5_K, QK_NL, dequantize_q5_K>;\n"
                                             "template [[host_name(\"kernel_get_rows_q6_K\")]] kernel get_rows_t kernel_get_rows<block_q6_K, QK_NL, dequantize_q6_K>;\n"
                                             "\n"
                                             "typedef void (mat_mm_t)(device const uchar *, device const float *, device float *, constant int64_t &,\\\n"
                                             "                             constant int64_t &, constant int64_t &, constant int64_t &, constant int64_t &, \\\n"
                                             "                             constant int64_t &, constant int64_t &, constant uint &, threadgroup uchar *, uint3, uint, uint);\n"
                                             "\n"
                                             "template [[host_name(\"kernel_mul_mm_f16_f32\")]]  kernel mat_mm_t kernel_mul_mm<half4x4,    1, dequantize_f16>;\n"
                                             "template [[host_name(\"kernel_mul_mm_q4_0_f32\")]] kernel mat_mm_t kernel_mul_mm<block_q4_0, 2, dequantize_q4_0>;\n"
                                             "template [[host_name(\"kernel_mul_mm_q4_1_f32\")]] kernel mat_mm_t kernel_mul_mm<block_q4_1, 2, dequantize_q4_1>;\n"
                                             "template [[host_name(\"kernel_mul_mm_q8_0_f32\")]] kernel mat_mm_t kernel_mul_mm<block_q8_0, 2, dequantize_q8_0>;\n"
                                             "template [[host_name(\"kernel_mul_mm_q2_K_f32\")]] kernel mat_mm_t kernel_mul_mm<block_q2_K, QK_NL, dequantize_q2_K>;\n"
                                             "template [[host_name(\"kernel_mul_mm_q3_K_f32\")]] kernel mat_mm_t kernel_mul_mm<block_q3_K, QK_NL, dequantize_q3_K>;\n"
                                             "template [[host_name(\"kernel_mul_mm_q4_K_f32\")]] kernel mat_mm_t kernel_mul_mm<block_q4_K, QK_NL, dequantize_q4_K>;\n"
                                             "template [[host_name(\"kernel_mul_mm_q5_K_f32\")]] kernel mat_mm_t kernel_mul_mm<block_q5_K, QK_NL, dequantize_q5_K>;\n"
                                             "template [[host_name(\"kernel_mul_mm_q6_K_f32\")]] kernel mat_mm_t kernel_mul_mm<block_q6_K, QK_NL, dequantize_q6_K>;";

// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end

struct ggml_metal_context * ggml_metal_init(int n_cb) {
    metal_printf("%s: allocating\n", __func__);

    id <MTLDevice> device;
    NSString * s;

#if TARGET_OS_OSX
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (device in devices) {
        s = [device name];
        metal_printf("%s: found device: %s\n", __func__, [s UTF8String]);
    }
#endif

    // Pick and show default Metal device
    device = MTLCreateSystemDefaultDevice();
    s = [device name];
    metal_printf("%s: picking default device: %s\n", __func__, [s UTF8String]);

    // Configure context
    struct ggml_metal_context * ctx = malloc(sizeof(struct ggml_metal_context));
    ctx->device = device;
    ctx->n_cb   = MIN(n_cb, GGML_METAL_MAX_BUFFERS);
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->n_buffers = 0;
    ctx->concur_list_len = 0;

    ctx->d_queue = dispatch_queue_create("llama.cpp", DISPATCH_QUEUE_CONCURRENT);

#ifdef GGML_SWIFT
    // load the default.metallib file
    {
        NSError * error = nil;

        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
        NSString * llamaBundlePath = [bundle pathForResource:@"llama_llama" ofType:@"bundle"];
        NSBundle * llamaBundle = [NSBundle bundleWithPath:llamaBundlePath];
        NSString * libPath = [llamaBundle pathForResource:@"default" ofType:@"metallib"];
        NSURL * libURL = [NSURL fileURLWithPath:libPath];

        // Load the metallib file into a Metal library
        ctx->library = [ctx->device newLibraryWithURL:libURL error:&error];

        if (error) {
            metal_printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }
    }
#else
    UNUSED(msl_library_source);

    // read the source from "ggml-metal.metal" into a string and use newLibraryWithSource
    {
        NSError * error = nil;

        //NSString * path = [[NSBundle mainBundle] pathForResource:@"../../examples/metal/metal" ofType:@"metal"];
        //NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
        //NSString * path = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
        //metal_printf("%s: loading '%s'\n", __func__, [path UTF8String]);

        NSString * src  = msl_library_source;

#ifdef GGML_QKK_64
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.preprocessorMacros = @{ @"QK_K" : @(64) };
        ctx->library = [ctx->device newLibraryWithSource:src options:options error:&error];
#else
        ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&error];
#endif
        if (error) {
            metal_printf("%s: error: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }
    }
#endif

    // load kernels
    {
        NSError * error = nil;
#define GGML_METAL_ADD_KERNEL(name) \
        ctx->function_##name = [ctx->library newFunctionWithName:@"kernel_"#name]; \
        ctx->pipeline_##name = [ctx->device newComputePipelineStateWithFunction:ctx->function_##name error:&error]; \
        metal_printf("%s: loaded %-32s %16p | th_max = %4d | th_width = %4d\n", __func__, "kernel_"#name, (void *) ctx->pipeline_##name, \
                (int) ctx->pipeline_##name.maxTotalThreadsPerThreadgroup, \
                (int) ctx->pipeline_##name.threadExecutionWidth); \
        if (error) { \
            metal_printf("%s: load pipeline error: %s\n", __func__, [[error description] UTF8String]); \
            return NULL; \
        }

        GGML_METAL_ADD_KERNEL(add);
        GGML_METAL_ADD_KERNEL(add_row);
        GGML_METAL_ADD_KERNEL(mul);
        GGML_METAL_ADD_KERNEL(mul_row);
        GGML_METAL_ADD_KERNEL(scale);
        GGML_METAL_ADD_KERNEL(silu);
        GGML_METAL_ADD_KERNEL(relu);
        GGML_METAL_ADD_KERNEL(gelu);
        GGML_METAL_ADD_KERNEL(soft_max);
        GGML_METAL_ADD_KERNEL(soft_max_4);
        GGML_METAL_ADD_KERNEL(diag_mask_inf);
        GGML_METAL_ADD_KERNEL(diag_mask_inf_8);
        GGML_METAL_ADD_KERNEL(get_rows_f16);
        GGML_METAL_ADD_KERNEL(get_rows_q4_0);
        GGML_METAL_ADD_KERNEL(get_rows_q4_1);
        GGML_METAL_ADD_KERNEL(get_rows_q8_0);
        GGML_METAL_ADD_KERNEL(get_rows_q2_K);
        GGML_METAL_ADD_KERNEL(get_rows_q3_K);
        GGML_METAL_ADD_KERNEL(get_rows_q4_K);
        GGML_METAL_ADD_KERNEL(get_rows_q5_K);
        GGML_METAL_ADD_KERNEL(get_rows_q6_K);
        GGML_METAL_ADD_KERNEL(rms_norm);
        GGML_METAL_ADD_KERNEL(norm);
        GGML_METAL_ADD_KERNEL(mul_mat_f16_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_f16_f32_1row);
        GGML_METAL_ADD_KERNEL(mul_mat_f16_f32_l4);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_0_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_1_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q8_0_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q2_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q3_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q5_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q6_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_f16_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q4_0_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q8_0_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q4_1_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q2_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q3_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q4_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q5_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q6_K_f32);
        GGML_METAL_ADD_KERNEL(rope);
        GGML_METAL_ADD_KERNEL(alibi_f32);
        GGML_METAL_ADD_KERNEL(cpy_f32_f16);
        GGML_METAL_ADD_KERNEL(cpy_f32_f32);
        GGML_METAL_ADD_KERNEL(cpy_f16_f16);

#undef GGML_METAL_ADD_KERNEL
    }

    metal_printf("%s: hasUnifiedMemory              = %s\n",       __func__, ctx->device.hasUnifiedMemory ? "true" : "false");
#if TARGET_OS_OSX
    metal_printf("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, ctx->device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);
    if (ctx->device.maxTransferRate != 0) {
        metal_printf("%s: maxTransferRate               = %8.2f MB/s\n", __func__, ctx->device.maxTransferRate / 1024.0 / 1024.0);
    } else {
        metal_printf("%s: maxTransferRate               = built-in GPU\n", __func__);
    }
#endif

    return ctx;
}

void ggml_metal_free(struct ggml_metal_context * ctx) {
    metal_printf("%s: deallocating\n", __func__);
#define GGML_METAL_DEL_KERNEL(name) \
    [ctx->function_##name release]; \
    [ctx->pipeline_##name release];

    GGML_METAL_DEL_KERNEL(add);
    GGML_METAL_DEL_KERNEL(add_row);
    GGML_METAL_DEL_KERNEL(mul);
    GGML_METAL_DEL_KERNEL(mul_row);
    GGML_METAL_DEL_KERNEL(scale);
    GGML_METAL_DEL_KERNEL(silu);
    GGML_METAL_DEL_KERNEL(relu);
    GGML_METAL_DEL_KERNEL(gelu);
    GGML_METAL_DEL_KERNEL(soft_max);
    GGML_METAL_DEL_KERNEL(soft_max_4);
    GGML_METAL_DEL_KERNEL(diag_mask_inf_8);
    GGML_METAL_DEL_KERNEL(get_rows_f16);
    GGML_METAL_DEL_KERNEL(get_rows_q4_0);
    GGML_METAL_DEL_KERNEL(get_rows_q4_1);
    GGML_METAL_DEL_KERNEL(get_rows_q8_0);
    GGML_METAL_DEL_KERNEL(get_rows_q2_K);
    GGML_METAL_DEL_KERNEL(get_rows_q3_K);
    GGML_METAL_DEL_KERNEL(get_rows_q4_K);
    GGML_METAL_DEL_KERNEL(get_rows_q5_K);
    GGML_METAL_DEL_KERNEL(get_rows_q6_K);
    GGML_METAL_DEL_KERNEL(rms_norm);
    GGML_METAL_DEL_KERNEL(norm);
    GGML_METAL_DEL_KERNEL(mul_mat_f16_f32);
    GGML_METAL_DEL_KERNEL(mul_mat_f16_f32_1row);
    GGML_METAL_DEL_KERNEL(mul_mat_f16_f32_l4);
    GGML_METAL_DEL_KERNEL(mul_mat_q4_0_f32);
    GGML_METAL_DEL_KERNEL(mul_mat_q4_1_f32);
    GGML_METAL_DEL_KERNEL(mul_mat_q8_0_f32);
    GGML_METAL_DEL_KERNEL(mul_mat_q2_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mat_q3_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mat_q4_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mat_q5_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mat_q6_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_f16_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_q4_0_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_q8_0_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_q4_1_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_q2_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_q3_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_q4_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_q5_K_f32);
    GGML_METAL_DEL_KERNEL(mul_mm_q6_K_f32);
    GGML_METAL_DEL_KERNEL(rope);
    GGML_METAL_DEL_KERNEL(alibi_f32);
    GGML_METAL_DEL_KERNEL(cpy_f32_f16);
    GGML_METAL_DEL_KERNEL(cpy_f32_f32);
    GGML_METAL_DEL_KERNEL(cpy_f16_f16);

#undef GGML_METAL_DEL_KERNEL

    for (int i = 0; i < ctx->n_buffers; ++i) {
        [ctx->buffers[i].metal release];
    }

    [ctx->library release];
    [ctx->queue release];
    [ctx->device release];

    dispatch_release(ctx->d_queue);

    free(ctx);
}

void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        metal_printf("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }

    return data;
}

void ggml_metal_host_free(void * data) {
    free(data);
}

void ggml_metal_set_n_cb(struct ggml_metal_context * ctx, int n_cb) {
    ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_BUFFERS);
}

int ggml_metal_if_optimized(struct ggml_metal_context * ctx) {
    return ctx->concur_list_len;
}

int * ggml_metal_get_concur_list(struct ggml_metal_context * ctx) {
    return ctx->concur_list;
}

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
static id<MTLBuffer> ggml_metal_get_buffer(struct ggml_metal_context * ctx, struct ggml_tensor * t, size_t * offs) {
    //metal_printf("%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) ctx->buffers[i].data;

        if (ioffs >= 0 && ioffs + tsize <= (int64_t) ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            //metal_printf("%s: '%s' tensor '%16s', offs = %8ld\n", __func__, ctx->buffers[i].name, t->name, *offs);

            return ctx->buffers[i].metal;
        }
    }

    metal_printf("%s: error: buffer is nil\n", __func__);

    return nil;
}

bool ggml_metal_add_buffer(
        struct ggml_metal_context * ctx,
                     const char * name,
                           void * data,
                         size_t   size,
                         size_t   max_size) {
    if (ctx->n_buffers >= GGML_METAL_MAX_BUFFERS) {
        metal_printf("%s: too many buffers\n", __func__);
        return false;
    }

    if (data) {
        // verify that the buffer does not overlap with any of the existing buffers
        for (int i = 0; i < ctx->n_buffers; ++i) {
            const int64_t ioffs = (int64_t) data - (int64_t) ctx->buffers[i].data;

            if (ioffs >= 0 && ioffs < (int64_t) ctx->buffers[i].size) {
                metal_printf("%s: error: buffer '%s' overlaps with '%s'\n", __func__, name, ctx->buffers[i].name);
                return false;
            }
        }

        const size_t size_page = sysconf(_SC_PAGESIZE);

        size_t size_aligned = size;
        if ((size_aligned % size_page) != 0) {
            size_aligned += (size_page - (size_aligned % size_page));
        }

        // the buffer fits into the max buffer size allowed by the device
        if (size_aligned <= ctx->device.maxBufferLength) {
            ctx->buffers[ctx->n_buffers].name = name;
            ctx->buffers[ctx->n_buffers].data = data;
            ctx->buffers[ctx->n_buffers].size = size;

            ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:data length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (ctx->buffers[ctx->n_buffers].metal == nil) {
                metal_printf("%s: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_aligned / 1024.0 / 1024.0);
                return false;
            }

            metal_printf("%s: allocated '%-16s' buffer, size = %8.2f MB", __func__, name, size_aligned / 1024.0 / 1024.0);

            ++ctx->n_buffers;
        } else {
            // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
            // one of the views
            const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
            const size_t size_step = ctx->device.maxBufferLength - size_ovlp;
            const size_t size_view = ctx->device.maxBufferLength;

            for (size_t i = 0; i < size; i += size_step) {
                const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

                ctx->buffers[ctx->n_buffers].name = name;
                ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
                ctx->buffers[ctx->n_buffers].size = size_step_aligned;

                ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:(void *) ((uint8_t *) data + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (ctx->buffers[ctx->n_buffers].metal == nil) {
                    metal_printf("%s: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }

                metal_printf("%s: allocated '%-16s' buffer, size = %8.2f MB, offs = %12ld", __func__, name, size_step_aligned / 1024.0 / 1024.0, i);
                if (i + size_step < size) {
                    metal_printf("\n");
                }

                ++ctx->n_buffers;
            }
        }

#if TARGET_OS_OSX
        metal_printf(", (%8.2f / %8.2f)",
                ctx->device.currentAllocatedSize / 1024.0 / 1024.0,
                ctx->device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (ctx->device.currentAllocatedSize > ctx->device.recommendedMaxWorkingSetSize) {
            metal_printf(", warning: current allocated size is greater than the recommended max working set size\n");
        } else {
            metal_printf("\n");
        }
#else
        metal_printf(", (%8.2f)\n", ctx->device.currentAllocatedSize / 1024.0 / 1024.0);
#endif
    }

    return true;
}

void ggml_metal_set_tensor(
        struct ggml_metal_context * ctx,
        struct ggml_tensor * t) {
    size_t offs;
    id<MTLBuffer> id_dst = ggml_metal_get_buffer(ctx, t, &offs);

    memcpy((void *) ((uint8_t *) id_dst.contents + offs), t->data, ggml_nbytes(t));
}

void ggml_metal_get_tensor(
        struct ggml_metal_context * ctx,
        struct ggml_tensor * t) {
    size_t offs;
    id<MTLBuffer> id_src = ggml_metal_get_buffer(ctx, t, &offs);

    memcpy(t->data, (void *) ((uint8_t *) id_src.contents + offs), ggml_nbytes(t));
}

void ggml_metal_graph_find_concurrency(
        struct ggml_metal_context * ctx,
        struct ggml_cgraph * gf, bool check_mem) {
    int search_depth = gf->n_nodes; //we only find concurrency in this range to avoid wasting too much time
    int nodes_unused[GGML_MAX_CONCUR];

    for (int i = 0; i < GGML_MAX_CONCUR; i++) { ctx->concur_list[i] = 0; }
    for (int i = 0; i < gf->n_nodes;     i++) { nodes_unused[i]     = 1; }
    ctx->concur_list_len = 0;

    int n_left    = gf->n_nodes;
    int n_start   = 0; // all nodes before n_start at nodes_unused array have been sorted and store back to ctx->concur_list
    int level_pos = 0; // at ctx->concur_list, the last layer (level) ends at level_pos

    while (n_left > 0) {
        // number of nodes at a layer (that can be issued concurrently)
        int concurrency = 0;
        for (int i = n_start; i < ((n_start + search_depth > gf->n_nodes) ? gf->n_nodes : n_start + search_depth); i++) {
            if (nodes_unused[i]) {
                // if the requirements for gf->nodes[i] are satisfied
                int exe_flag = 1;

                // scan all srcs
                for (int src_ind = 0; src_ind < GGML_MAX_SRC; src_ind++) {
                    struct ggml_tensor * src_cur = gf->nodes[i]->src[src_ind];
                    if (src_cur) {
                        // if is leaf nodes it's satisfied.
                        // TODO: ggml_is_leaf()
                        if (src_cur->op == GGML_OP_NONE && src_cur->grad == NULL) {
                            continue;
                        }

                        // otherwise this src should be the output from previous nodes.
                        int is_found = 0;

                        // scan 2*search_depth back because we inserted barrier.
                        //for (int j = ((level_pos - 2*search_depth) < 0 ? 0 : (level_pos - 2*search_depth)); j < level_pos; j++) {
                        for (int j = MAX(0, level_pos - 2*search_depth); j < level_pos; j++) {
                            if (ctx->concur_list[j] >= 0 && gf->nodes[ctx->concur_list[j]] == src_cur) {
                                is_found = 1;
                                break;
                            }
                        }
                        if (is_found == 0) {
                            exe_flag = 0;
                            break;
                        }
                    }
                }
                if (exe_flag && check_mem) {
                    // check if nodes[i]'s data will be overwritten by a node before nodes[i].
                    // if node[5] and node[3] write to the same memory region, then we can't issue node[5] before node[3]
                    int64_t data_start = (int64_t) gf->nodes[i]->data;
                    int64_t length     = (int64_t) ggml_nbytes(gf->nodes[i]);
                    for (int j = n_start; j < i; j++) {
                        if (nodes_unused[j] && gf->nodes[j]->op != GGML_OP_RESHAPE \
                                            && gf->nodes[j]->op != GGML_OP_VIEW \
                                            && gf->nodes[j]->op != GGML_OP_TRANSPOSE \
                                            && gf->nodes[j]->op != GGML_OP_PERMUTE) {
                            if (((int64_t)gf->nodes[j]->data) >= data_start + length || \
                                ((int64_t)gf->nodes[j]->data) + (int64_t) ggml_nbytes(gf->nodes[j]) <= data_start) {
                                continue;
                            }

                            exe_flag = 0;
                        }
                    }
                }
                if (exe_flag) {
                    ctx->concur_list[level_pos + concurrency] = i;
                    nodes_unused[i] = 0;
                    concurrency++;
                    ctx->concur_list_len++;
                }
            }
        }
        n_left -= concurrency;
        // adding a barrier different layer
        ctx->concur_list[level_pos + concurrency] = -1;
        ctx->concur_list_len++;
        // jump all sorted nodes at nodes_bak
        while (!nodes_unused[n_start]) {
            n_start++;
        }
        level_pos += concurrency + 1;
    }

    if (ctx->concur_list_len > GGML_MAX_CONCUR) {
        metal_printf("%s: too many elements for metal ctx->concur_list!\n", __func__);
    }
}

void ggml_metal_graph_compute(
        struct ggml_metal_context * ctx,
               struct ggml_cgraph * gf) {
    @autoreleasepool {

    // if there is ctx->concur_list, dispatch concurrently
    // else fallback to serial dispatch
    MTLComputePassDescriptor * edesc = MTLComputePassDescriptor.computePassDescriptor;

    const bool has_concur = ctx->concur_list_len && ctx->concur_list_len <= GGML_MAX_CONCUR;

    const int n_nodes  = has_concur ? ctx->concur_list_len      : gf->n_nodes;
    edesc.dispatchType = has_concur ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;

    // create multiple command buffers and enqueue them
    // then, we encode the graph into the command buffers in parallel

    const int n_cb = ctx->n_cb;

    for (int i = 0; i < n_cb; ++i) {
        ctx->command_buffers[i] = [ctx->queue commandBuffer];

        // enqueue the command buffers in order to specify their execution order
        [ctx->command_buffers[i] enqueue];

        ctx->command_encoders[i] = [ctx->command_buffers[i] computeCommandEncoderWithDescriptor: edesc];
    }

    for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
        const int n_nodes_per_cb = (n_nodes + n_cb - 1) / n_cb;

        dispatch_async(ctx->d_queue, ^{
            size_t offs_src0 = 0;
            size_t offs_src1 = 0;
            size_t offs_dst  = 0;

            id<MTLCommandBuffer> command_buffer  = ctx->command_buffers[cb_idx];
            id<MTLComputeCommandEncoder> encoder = ctx->command_encoders[cb_idx];

            const int node_start =                                      (cb_idx + 0) * n_nodes_per_cb;
            const int node_end   = MIN((cb_idx == n_cb - 1) ? n_nodes : (cb_idx + 1) * n_nodes_per_cb, n_nodes);

            for (int ind = node_start; ind < node_end; ++ind) {
                const int i = has_concur ? ctx->concur_list[ind] : ind;

                if (i == -1) {
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    continue;
                }

                //metal_printf("%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

                struct ggml_tensor * src0 = gf->nodes[i]->src[0];
                struct ggml_tensor * src1 = gf->nodes[i]->src[1];
                struct ggml_tensor * dst  = gf->nodes[i];

                const int64_t  ne00 = src0 ? src0->ne[0] : 0;
                const int64_t  ne01 = src0 ? src0->ne[1] : 0;
                const int64_t  ne02 = src0 ? src0->ne[2] : 0;
                const int64_t  ne03 = src0 ? src0->ne[3] : 0;

                const uint64_t nb00 = src0 ? src0->nb[0] : 0;
                const uint64_t nb01 = src0 ? src0->nb[1] : 0;
                const uint64_t nb02 = src0 ? src0->nb[2] : 0;
                const uint64_t nb03 = src0 ? src0->nb[3] : 0;

                const int64_t  ne10 = src1 ? src1->ne[0] : 0;
                const int64_t  ne11 = src1 ? src1->ne[1] : 0;
                const int64_t  ne12 = src1 ? src1->ne[2] : 0;
                const int64_t  ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

                const uint64_t nb10 = src1 ? src1->nb[0] : 0;
                const uint64_t nb11 = src1 ? src1->nb[1] : 0;
                const uint64_t nb12 = src1 ? src1->nb[2] : 0;
                const uint64_t nb13 = src1 ? src1->nb[3] : 0; UNUSED(nb13);

                const int64_t  ne0  = dst ? dst->ne[0] : 0;
                const int64_t  ne1  = dst ? dst->ne[1] : 0;
                const int64_t  ne2  = dst ? dst->ne[2] : 0;
                const int64_t  ne3  = dst ? dst->ne[3] : 0;

                const uint64_t nb0  = dst ? dst->nb[0] : 0;
                const uint64_t nb1  = dst ? dst->nb[1] : 0;
                const uint64_t nb2  = dst ? dst->nb[2] : 0;
                const uint64_t nb3  = dst ? dst->nb[3] : 0;

                const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
                const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
                const enum ggml_type dstt  = dst  ? dst->type  : GGML_TYPE_COUNT;

                id<MTLBuffer> id_src0 = src0 ? ggml_metal_get_buffer(ctx, src0, &offs_src0) : nil;
                id<MTLBuffer> id_src1 = src1 ? ggml_metal_get_buffer(ctx, src1, &offs_src1) : nil;
                id<MTLBuffer> id_dst  = dst  ? ggml_metal_get_buffer(ctx, dst,  &offs_dst)  : nil;

                //metal_printf("%s: op - %s\n", __func__, ggml_op_name(dst->op));
                //if (src0) {
                //    metal_printf("%s: src0 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src0t), ne00, ne01, ne02,
                //            ggml_is_contiguous(src0), src0->name);
                //}
                //if (src1) {
                //    metal_printf("%s: src1 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src1t), ne10, ne11, ne12,
                //            ggml_is_contiguous(src1), src1->name);
                //}
                //if (dst) {
                //    metal_printf("%s: dst  - %4s [%5lld, %5lld, %5lld], 1, %s\n",  __func__, ggml_type_name(dstt),  ne0,  ne1,  ne2,
                //            dst->name);
                //}

                switch (dst->op) {
                    case GGML_OP_NONE:
                    case GGML_OP_RESHAPE:
                    case GGML_OP_VIEW:
                    case GGML_OP_TRANSPOSE:
                    case GGML_OP_PERMUTE:
                        {
                            // noop
                        } break;
                    case GGML_OP_ADD:
                        {
                            GGML_ASSERT(ggml_is_contiguous(src0));

                            // utilize float4
                            GGML_ASSERT(ne00 % 4 == 0);
                            const int64_t nb = ne00/4;

                            if (ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                [encoder setComputePipelineState:ctx->pipeline_add_row];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_add];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&nb     length:sizeof(nb) atIndex:3];

                            const int64_t n = ggml_nelements(dst)/4;

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_MUL:
                        {
                            GGML_ASSERT(ggml_is_contiguous(src0));

                            // utilize float4
                            GGML_ASSERT(ne00 % 4 == 0);
                            const int64_t nb = ne00/4;

                            if (ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                [encoder setComputePipelineState:ctx->pipeline_mul_row];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_mul];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&nb     length:sizeof(nb) atIndex:3];

                            const int64_t n = ggml_nelements(dst)/4;

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_SCALE:
                        {
                            const float scale = *(const float *) src1->data;

                            [encoder setComputePipelineState:ctx->pipeline_scale];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

                            const int64_t n = ggml_nelements(dst)/4;

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_UNARY:
                        switch (ggml_get_unary_op(gf->nodes[i])) {
                            case GGML_UNARY_OP_SILU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_silu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst)/4;

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case GGML_UNARY_OP_RELU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_relu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case GGML_UNARY_OP_GELU:
                                {
                                    [encoder setComputePipelineState:ctx->pipeline_gelu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst)/4;

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            default:
                                {
                                    metal_printf("%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                                    GGML_ASSERT(false);
                                }
                        } break;
                    case GGML_OP_SOFT_MAX:
                        {
                            const int nth = 32;

                            if (ne00%4 == 0) {
                                [encoder setComputePipelineState:ctx->pipeline_soft_max_4];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_soft_max];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:4];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_DIAG_MASK_INF:
                        {
                            const int n_past = ((int32_t *)(dst->op_params))[0];

                            if (ne00%8 == 0) {
                                [encoder setComputePipelineState:ctx->pipeline_diag_mask_inf_8];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_diag_mask_inf];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00   length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01   length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&n_past length:sizeof(int)  atIndex:4];

                            if (ne00%8 == 0) {
                                [encoder dispatchThreadgroups:MTLSizeMake(ne00*ne01*ne02/8, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            }
                            else {
                                [encoder dispatchThreadgroups:MTLSizeMake(ne00, ne01, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            }
                        } break;
                    case GGML_OP_MUL_MAT:
                        {
                            // TODO: needs to be updated after PR: https://github.com/ggerganov/ggml/pull/224

                            GGML_ASSERT(ne00 == ne10);
                            // GGML_ASSERT(ne02 == ne12); // Should be checked on individual data types until broadcast is implemented everywhere
                            uint gqa = ne12/ne02;
                            GGML_ASSERT(ne03 == ne13);

                            // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                            // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                            if (ggml_is_contiguous(src0) &&
                                ggml_is_contiguous(src1) &&
                                src1t == GGML_TYPE_F32 &&
                                [ctx->device supportsFamily:MTLGPUFamilyApple7] &&
                                ne00%32 == 0 &&
                                ne11 > 1) {
                                switch (src0->type) {
                                    case GGML_TYPE_F16:  [encoder setComputePipelineState:ctx->pipeline_mul_mm_f16_f32];  break;
                                    case GGML_TYPE_Q4_0: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q4_0_f32]; break;
                                    case GGML_TYPE_Q4_1: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q4_1_f32]; break;
                                    case GGML_TYPE_Q8_0: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q8_0_f32]; break;
                                    case GGML_TYPE_Q2_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q2_K_f32]; break;
                                    case GGML_TYPE_Q3_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q3_K_f32]; break;
                                    case GGML_TYPE_Q4_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q4_K_f32]; break;
                                    case GGML_TYPE_Q5_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q5_K_f32]; break;
                                    case GGML_TYPE_Q6_K: [encoder setComputePipelineState:ctx->pipeline_mul_mm_q6_K_f32]; break;
                                    default: GGML_ASSERT(false && "MUL MAT-MAT not implemented");
                                }
                                [encoder setBuffer:id_src0 offset:offs_src0    atIndex:0];
                                [encoder setBuffer:id_src1 offset:offs_src1    atIndex:1];
                                [encoder setBuffer:id_dst  offset:offs_dst     atIndex:2];
                                [encoder setBytes:&ne00    length:sizeof(ne00) atIndex:3];
                                [encoder setBytes:&ne02    length:sizeof(ne02) atIndex:4];
                                [encoder setBytes:&nb01    length:sizeof(nb01) atIndex:5];
                                [encoder setBytes:&nb02    length:sizeof(nb02) atIndex:6];
                                [encoder setBytes:&ne12    length:sizeof(ne12) atIndex:7];
                                [encoder setBytes:&ne0     length:sizeof(ne0)  atIndex:8];
                                [encoder setBytes:&ne1     length:sizeof(ne1)  atIndex:9];
                                [encoder setBytes:&gqa     length:sizeof(gqa)  atIndex:10];
                                [encoder setThreadgroupMemoryLength:8192 atIndex:0];
                                [encoder dispatchThreadgroups:MTLSizeMake( (ne11+31)/32, (ne01+63) / 64, ne12) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                            } else {
                                int nth0 = 32;
                                int nth1 = 1;
                                int nrows = 1;

                                // use custom matrix x vector kernel
                                switch (src0t) {
                                    case GGML_TYPE_F16:
                                        {
                                            nth0 = 32;
                                            nth1 = 1;
                                            if (ne11 * ne12 < 4) {
                                                [encoder setComputePipelineState:ctx->pipeline_mul_mat_f16_f32_1row];
                                            } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                                                [encoder setComputePipelineState:ctx->pipeline_mul_mat_f16_f32_l4];
                                                nrows = ne11;
                                            } else {
                                                [encoder setComputePipelineState:ctx->pipeline_mul_mat_f16_f32];
                                                nrows = 4;
                                            }
                                        } break;
                                    case GGML_TYPE_Q4_0:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_0_f32];
                                        } break;
                                    case GGML_TYPE_Q4_1:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_1_f32];
                                        } break;
                                    case GGML_TYPE_Q8_0:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q8_0_f32];
                                        } break;
                                    case GGML_TYPE_Q2_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q2_K_f32];
                                        } break;
                                    case GGML_TYPE_Q3_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q3_K_f32];
                                        } break;
                                    case GGML_TYPE_Q4_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 4; //1;
                                            nth1 = 8; //32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_K_f32];
                                        } break;
                                    case GGML_TYPE_Q5_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q5_K_f32];
                                        } break;
                                    case GGML_TYPE_Q6_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q6_K_f32];
                                        } break;
                                    default:
                                        {
                                            metal_printf("Asserting on type %d\n",(int)src0t);
                                            GGML_ASSERT(false && "not implemented");
                                        }
                                };

                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                                [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                                [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:4];
                                [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:5];
                                [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:6];
                                [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:7];
                                [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:8];
                                [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:9];
                                [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:10];
                                [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:11];
                                [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:12];
                                [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:13];
                                [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:14];
                                [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:15];
                                [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:16];
                                [encoder setBytes:&gqa  length:sizeof(gqa)  atIndex:17];

                                if (src0t == GGML_TYPE_Q4_0 || src0t == GGML_TYPE_Q4_1 || src0t == GGML_TYPE_Q8_0 ||
                                    src0t == GGML_TYPE_Q2_K) {// || src0t == GGML_TYPE_Q4_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == GGML_TYPE_Q4_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == GGML_TYPE_Q3_K) {
#ifdef GGML_QKK_64
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#else
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#endif
                                }
                                else if (src0t == GGML_TYPE_Q5_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == GGML_TYPE_Q6_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                } else {
                                    int64_t ny = (ne11 + nrows - 1)/nrows;
                                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ny, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                            }
                        } break;
                    case GGML_OP_GET_ROWS:
                        {
                            switch (src0->type) {
                                case GGML_TYPE_F16:  [encoder setComputePipelineState:ctx->pipeline_get_rows_f16];  break;
                                case GGML_TYPE_Q4_0: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_0]; break;
                                case GGML_TYPE_Q4_1: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_1]; break;
                                case GGML_TYPE_Q8_0: [encoder setComputePipelineState:ctx->pipeline_get_rows_q8_0]; break;
                                case GGML_TYPE_Q2_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q2_K]; break;
                                case GGML_TYPE_Q3_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q3_K]; break;
                                case GGML_TYPE_Q4_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_K]; break;
                                case GGML_TYPE_Q5_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q5_K]; break;
                                case GGML_TYPE_Q6_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q6_K]; break;
                                default: GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&(src0->ne[0]) length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&(src0->nb[1]) length:sizeof(uint64_t) atIndex:4];
                            [encoder setBytes:&(dst->nb[1])  length:sizeof(uint64_t) atIndex:5];

                            const int64_t n = ggml_nelements(src1);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_RMS_NORM:
                        {
                            float eps;
                            memcpy(&eps, dst->op_params, sizeof(float));

                            const int nth = 512;

                            [encoder setComputePipelineState:ctx->pipeline_rms_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps  length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth/32*sizeof(float) atIndex:0];

                            const int64_t nrows = ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_NORM:
                        {
                            float eps;
                            memcpy(&eps, dst->op_params, sizeof(float));

                            const int nth = 256;

                            [encoder setComputePipelineState:ctx->pipeline_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps     length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth*sizeof(float) atIndex:0];

                            const int64_t nrows = ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_ALIBI:
                        {
                            GGML_ASSERT((src0t == GGML_TYPE_F32));

                            const int n_past = ((int32_t *) dst->op_params)[0]; UNUSED(n_past);
                            const int n_head = ((int32_t *) dst->op_params)[1];
                            float max_bias;
                            memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

                            if (__builtin_popcount(n_head) != 1) {
                                GGML_ASSERT(false && "only power-of-two n_head implemented");
                            }

                            const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));
                            const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);

                            [encoder setComputePipelineState:ctx->pipeline_alibi_f32];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03 length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00 length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02 length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03 length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0  length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1  length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2  length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3  length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0  length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1  length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2  length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3  length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&m0  length:sizeof(    float) atIndex:18];

                            const int nth = 32;

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_ROPE:
                        {
                            const int n_past = ((int32_t *) dst->op_params)[0];
                            const int n_dims = ((int32_t *) dst->op_params)[1];
                            const int mode   = ((int32_t *) dst->op_params)[2];

                            float freq_base;
                            float freq_scale;
                            memcpy(&freq_base,  (int32_t *) dst->op_params + 4, sizeof(float));
                            memcpy(&freq_scale, (int32_t *) dst->op_params + 5, sizeof(float));

                            [encoder setComputePipelineState:ctx->pipeline_rope];
                            [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01    length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02    length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03    length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00    length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02    length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03    length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0     length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1     length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2     length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3     length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0     length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1     length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2     length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3     length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&n_past  length:sizeof(     int) atIndex:18];
                            [encoder setBytes:&n_dims  length:sizeof(     int) atIndex:19];
                            [encoder setBytes:&mode    length:sizeof(     int) atIndex:20];
                            [encoder setBytes:&freq_base  length:sizeof(float) atIndex:21];
                            [encoder setBytes:&freq_scale length:sizeof(float) atIndex:22];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                        } break;
                    case GGML_OP_DUP:
                    case GGML_OP_CPY:
                    case GGML_OP_CONT:
                        {
                            const int nth = 32;

                            switch (src0t) {
                                case GGML_TYPE_F32:
                                    {
                                        switch (dstt) {
                                            case GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f16]; break;
                                            case GGML_TYPE_F32: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f32]; break;
                                            default: GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                case GGML_TYPE_F16:
                                    {
                                        switch (dstt) {
                                            case GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f16_f16]; break;
                                            case GGML_TYPE_F32: GGML_ASSERT(false && "cpy_f16_f32 not implemented"); break;
                                            default: GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                default: GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01    length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02    length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03    length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00    length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02    length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03    length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0     length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1     length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2     length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3     length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0     length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1     length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2     length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3     length:sizeof(uint64_t) atIndex:17];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    default:
                        {
                            metal_printf("%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                            GGML_ASSERT(false);
                        }
                }
            }

            if (encoder != nil) {
                [encoder endEncoding];
                encoder = nil;
            }

            [command_buffer commit];
        });
    }

    // wait for all threads to finish
    dispatch_barrier_sync(ctx->d_queue, ^{});

    // check status of command buffers
    // needed to detect if the device ran out-of-memory for example (#1881)
    for (int i = 0; i < n_cb; i++) {
        [ctx->command_buffers[i] waitUntilCompleted];

        MTLCommandBufferStatus status = (MTLCommandBufferStatus) [ctx->command_buffers[i] status];
        if (status != MTLCommandBufferStatusCompleted) {
            metal_printf("%s: command buffer %d failed with status %lu\n", __func__, i, status);
            GGML_ASSERT(false);
        }
    }

    }
}
