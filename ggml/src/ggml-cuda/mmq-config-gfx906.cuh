// gfx906 (Vega 20): rows for the Bonsai types, everything else uses the RDNA2 table.
// PTQ1_0 below J = 64: the RDNA2 PQ2_0 rows (same tile layout).
// 512 threads at J = 64 keep J/nwarps <= 8, so the q8_0 dp4a vec dot reads y from VRAM (scalar loads).
#ifndef GFX906_MMQ_NT
#define GFX906_MMQ_NT 512
#endif
#ifndef GFX906_MMQ_I
#define GFX906_MMQ_I 64
#endif
#ifndef GFX906_MMQ_OCC
#define GFX906_MMQ_OCC 2
#endif
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_gfx906(ggml_type type, int J, bool fallback) {
    CASE(GGML_TYPE_PQ2_0, GFX906_MMQ_NT, GFX906_MMQ_OCC, GFX906_MMQ_I, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_PQ2_0, GFX906_MMQ_NT, GFX906_MMQ_OCC, GFX906_MMQ_I, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);

    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_PTQ1_0, GFX906_MMQ_NT, GFX906_MMQ_OCC, GFX906_MMQ_I, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_PTQ1_0, 256, 2, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_PTQ1_0, GFX906_MMQ_NT, GFX906_MMQ_OCC, GFX906_MMQ_I, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false);

    return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
}
