static void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    // some shaders have a minimum subgroup size
    const uint32_t subgroup_size_8 = std::max(device->subgroup_size, 8u);
    const uint32_t subgroup_size_16 = std::max(device->subgroup_size, 16u);
    const uint32_t subgroup_size_32 = std::max(device->subgroup_size, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    const uint32_t mm_warp_8  = std::min(subgroup_size_8,  64u);
    const uint32_t mm_warp_16 = std::min(subgroup_size_16, 64u);

    const uint32_t mul_mat_subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
    const uint32_t mul_mat_subgroup_size_8 = std::max(mul_mat_subgroup_size, 8u);
    const uint32_t mul_mat_subgroup_size_16 = std::max(mul_mat_subgroup_size, 16u);
    const uint32_t mul_mat_subgroup_size_32 = std::max(mul_mat_subgroup_size, 32u);
    const uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_size_8,  64u);
    const uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_size_16, 64u);

    const bool subgroup_min_size_16 = (!device->subgroup_size_control && device->subgroup_size >= 16) ||
                                      (device->subgroup_size_control && device->subgroup_max_size >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    static constexpr size_t WARP_SIZE_IDX = 10;
    std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                          l_warptile_id, m_warptile_id, s_warptile_id,
                          l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                          l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                          l_warptile_mmq_cm1_int, m_warptile_mmq_cm1_int, s_warptile_mmq_cm1_int,
                          l_warptile_mmq_cm1_int_k, m_warptile_mmq_cm1_int_k, s_warptile_mmq_cm1_int_k,
                          l_warptile_mmq_int_k, m_warptile_mmq_int_k, s_warptile_mmq_int_k,
                          l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                          l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid,
                          l_warptile_mmqid_int, m_warptile_mmqid_int, s_warptile_mmqid_int,
                          l_warptile_mmqid_int_k, m_warptile_mmqid_int_k, s_warptile_mmqid_int_k;
    std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                            l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                            l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                            l_mmq_cm1_wg_denoms_k, m_mmq_cm1_wg_denoms_k, s_mmq_cm1_wg_denoms_k,
                            l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

    uint32_t l_align, m_align, s_align;

    // RDNA3.5 preferred wave32 here
    const bool cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_size_control &&
                                device->subgroup_min_size <= 32 && device->subgroup_max_size >= 32;
    const uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_size;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    bool has_claimed_task = false;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_lock(device->compile_mutex);

    if (device->coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        l_warptile = { 256, 128, 256, 64, 1 };
        m_warptile = { 256, 128, 128, 64, 0 };
        s_warptile = { 128,  64,  64, 64, 0 };
        l_wg_denoms = {128, 256, 1 };
        m_wg_denoms = {128, 128, 1 };
        s_wg_denoms = { 64,  64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        l_warptile_mmq = { 256, 128, 256, 64, 1 };
        m_warptile_mmq = { 256, 128, 128, 64, 1 };
        s_warptile_mmq = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms = { 128, 256, 1 };
        m_mmq_wg_denoms = { 128, 128, 1 };
        s_mmq_wg_denoms = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        l_warptile_mmq_k = { 256, 128, 256, 64, 1 };
        m_warptile_mmq_k = { 256, 128, 128, 64, 1 };
        s_warptile_mmq_k = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms_k = { 128, 256, 1 };
        m_mmq_wg_denoms_k = { 128, 128, 1 };
        s_mmq_wg_denoms_k = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul_id
        const uint32_t mmqid_bk = device->coopmat2_decode_vector ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } else {
        // Matrix cores require different warp group sizes
        const uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        const uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        const uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        const uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        const uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        const uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        const uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        const uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        const uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        const uint32_t itm = device->coopmat_int_m;
        const uint32_t itn = device->coopmat_int_n;
        const uint32_t itk = device->coopmat_int_k;

        const uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_size_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_size_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_size_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_size_8 };

        const auto cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            return cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
        };

        l_warptile_mmq_cm1_int = { cm1_bs(128, 128), 128, 128, 32, std::min(cm1_sg, 128u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_warptile_mmq_cm1_int_k = { cm1_bs( 64, 128),  64, 128, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        m_warptile_mmq_cm1_int_k = { cm1_bs( 64,  64),  64,  64, 32, std::min(cm1_sg,  64u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };
        s_warptile_mmq_cm1_int_k = { cm1_bs( 32,  32),  32,  32, 32, std::min(cm1_sg,  32u), 32, 2, itm, itn, itk, cm1_sg, (uint32_t)device->architecture };

        l_mmq_cm1_wg_denoms_k = { l_warptile_mmq_cm1_int_k[1], l_warptile_mmq_cm1_int_k[2], 1 };
        m_mmq_cm1_wg_denoms_k = { m_warptile_mmq_cm1_int_k[1], m_warptile_mmq_cm1_int_k[2], 1 };
        s_mmq_cm1_wg_denoms_k = { s_warptile_mmq_cm1_int_k[1], s_warptile_mmq_cm1_int_k[2], 1 };

        // K-quants use even more registers, mitigate by setting WMITER to 1
        l_warptile_mmq_int_k = { 128,               128, 128, 32, mm_warp_8 * 2, 64, 1, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int_k = { 128,                64,  64, 32, mm_warp_8,     32, 1, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int_k = { subgroup_size_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_size_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_size_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_size_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_size_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_size_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_size_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_size_16 };

        // chip specific tuning
        if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } else if (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } else if (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        if (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            if (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } else {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        const bool use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        for (uint32_t i = 0; i < GGML_TYPE_COUNT; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            if (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, false, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::endl;
                throw std::runtime_error("Shared memory size too small for matrix multiplication.");
            } else if (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, false, t)) {
                device->mul_mat_m[i] = false;
                device->mul_mat_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, false, t)) {
                device->mul_mat_l[i] = false;
            }

            // Disable mul_mat_id if not enough shared memory is available
            if (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, true, t)) {
                device->mul_mat_id_s[i] = false;
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, true, t)) {
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, true, t)) {
                device->mul_mat_id_l[i] = false;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            const bool is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            const bool cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            const auto & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            const auto & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            const auto & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            const auto & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            const auto & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            const auto & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            const auto int_shmem_support = [&](const std::vector<uint32_t>& wt, bool id) {
                return use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            if (!int_shmem_support(s_int, false)) {
                device->mul_mat_s_int[i] = false;
                device->mul_mat_m_int[i] = false;
                device->mul_mat_l_int[i] = false;
            } else if (!int_shmem_support(m_int, false)) {
                device->mul_mat_m_int[i] = false;
                device->mul_mat_l_int[i] = false;
            } else if (!int_shmem_support(l_int, false)) {
                device->mul_mat_l_int[i] = false;
            }

            if (!int_shmem_support(s_intid, true)) {
                device->mul_mat_id_s_int[i] = false;
                device->mul_mat_id_m_int[i] = false;
                device->mul_mat_id_l_int[i] = false;
            } else if (!int_shmem_support(m_intid, true)) {
                device->mul_mat_id_m_int[i] = false;
                device->mul_mat_id_l_int[i] = false;
            } else if (!int_shmem_support(l_intid, true)) {
                device->mul_mat_id_l_int[i] = false;
            }
        }
    }

    auto const &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& base_pipeline, const char *name, size_t spv_size, const void* spv_data, const char *entrypoint,
                                              uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {

        if (!require_full_subgroups && required_subgroup_size == 0) {
            required_subgroup_size = get_subgroup_size(name, device->architecture);
        }

        vk_pipeline *ptr = &base_pipeline;

        int num_pipelines = 1;
#if defined(VK_EXT_shader_64bit_indexing)
        if (device->shader_64b_indexing) {
            num_pipelines = 2;
        }
#endif
        for (int i = 0; i < num_pipelines; ++i, ptr = &(*ptr)->next) {
            vk_pipeline &pipeline = *ptr;
            if (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            if (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_count = parameter_count;
                pipeline->push_constant_size = push_constant_size;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = true;
#if defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#endif
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            if (pipeline.get() != requested.get()) {
                continue;
            }

            if (pipeline->compiled) {
                continue;
            }

            wait_pipeline = pipeline;

            if (!pipeline->compile_pending) {
                pipeline->compile_pending = true;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_size = spv_size;
                claimed_task.spv_data = spv_data;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_count = parameter_count;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_full_subgroups = require_full_subgroups;
                claimed_task.required_subgroup_size = required_subgroup_size;
                has_claimed_task = true;
            }
        }
    };

    auto const &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, const std::string &name, size_t spv_size, const void* spv_data, const char *entrypoint,
                                              uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {
        return ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_size, spv_data, entrypoint,
                                       parameter_count, push_constant_size, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_full_subgroups, required_subgroup_size);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    for (auto &fa : device->pipeline_flash_attn_f32_f16) {
        if (fa.first.path != FA_SCALAR) continue;
        const uint32_t Br = fa.first.Br;
        const uint32_t Bc = fa.first.Bc;
        const bool aligned = fa.first.aligned;
        const bool f32acc = fa.first.f32acc;
        const uint32_t fa_sgs = fa.first.subgroup_size;
        const bool fa_ds = fa.first.subgroup_size == 0;

        const bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
        const bool use_mmq = ggml_vk_fa_scalar_uses_mmq(device, fa.first.k_type, fa.first.v_type);
        const void * spv_data = nullptr;
        size_t spv_size = 0;
        const char *name = nullptr;
        if (bf16_kv) {
            spv_data = flash_attn_f32_f16_fp32_data;
            spv_size = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } else if (use_mmq) {
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->fp16) {
                if (f32acc) { spv_data = flash_attn_f32_f16_int8_data;        spv_size = flash_attn_f32_f16_int8_len; }
                else        { spv_data = flash_attn_f32_f16_f16acc_int8_data; spv_size = flash_attn_f32_f16_f16acc_int8_len; }
            } else {
                spv_data = flash_attn_f32_f16_fp32_int8_data;
                spv_size = flash_attn_f32_f16_fp32_int8_len;
            }
#endif
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } else {
            if (device->fp16) {
                if (device->dot2_f16) {
                    if (f32acc) { spv_data = flash_attn_f32_f16_dot2_data;        spv_size = flash_attn_f32_f16_dot2_len; }
                    else        { spv_data = flash_attn_f32_f16_dot2_f16acc_data; spv_size = flash_attn_f32_f16_dot2_f16acc_len; }
                } else {
                    if (f32acc) { spv_data = flash_attn_f32_f16_data;        spv_size = flash_attn_f32_f16_len; }
                    else        { spv_data = flash_attn_f32_f16_f16acc_data; spv_size = flash_attn_f32_f16_f16acc_len; }
                }
            } else {
                spv_data = flash_attn_f32_f16_fp32_data;
                spv_size = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 8,
                                sizeof(vk_flash_attn_push_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.first), aligned ? Bc : 1, true,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat1_fa_support) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16) {
            if (fa.first.path != FA_COOPMAT1) continue;
            const uint32_t Br = fa.first.Br;
            const uint32_t Bc = fa.first.Bc;
            const bool aligned = fa.first.aligned;
            const bool f32acc = fa.first.f32acc;
            const uint32_t fa_sgs = fa.first.subgroup_size;
            const bool fa_ds = fa.first.subgroup_size == 0;

            const bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;

            const void * spv_data;
            size_t spv_size;
            const char *name;
            if (bf16_kv) {
#if defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                if (!device->coopmat_bf16_support) continue;
                spv_data = flash_attn_f32_f16_bf16_cm1_data;
                spv_size = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#else
                continue;
#endif
            } else {
                if (f32acc) { spv_data = flash_attn_f32_f16_cm1_data;        spv_size = flash_attn_f32_f16_cm1_len; }
                else        { spv_data = flash_attn_f32_f16_f16acc_cm1_data; spv_size = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 8,
                                    sizeof(vk_flash_attn_push_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, true,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#endif

#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16) {
            if (fa.first.path != FA_COOPMAT2) continue;
            const uint32_t Br = fa.first.Br;
            const uint32_t Bc = fa.first.Bc;
            const bool aligned = fa.first.aligned;
            const bool f32acc = fa.first.f32acc;

            const bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
            const void * spv_data;
            size_t spv_size;
            const char * name;
            if (bf16_kv) {
#if defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                if (!device->coopmat2_bf16_support) continue;
                spv_data = flash_attn_f32_f16_bf16_cm2_data;
                spv_size = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#else
                continue;
#endif
            } else if (aligned) {
                if (f32acc) { spv_data = flash_attn_f32_f16_cm2_data;        spv_size = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                else        { spv_data = flash_attn_f32_f16_f16acc_cm2_data; spv_size = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } else {
                if (f32acc) { spv_data = flash_attn_f32_f16_cm2_data;        spv_size = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                else        { spv_data = flash_attn_f32_f16_f16acc_cm2_data; spv_size = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 8,
                                    sizeof(vk_flash_attn_push_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, true, false, 0);
        }
    }
#endif

    auto const &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, bool aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        if (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        return spec;
    };

    auto const &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, bool aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        if (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        return spec;
    };

    static const ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#define FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#define FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#define FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    const int mul_mat_id_param_count = 5;

    using spec_fn_t = std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, bool)>;
    auto const &create_mm_pipelines = [&](
        const vk_matmul_pipeline_key& key,
        const std::vector<vk_tile_config>& tile_configs,
        const std::string& shader_name, size_t spv_len, const void* spv_data,
        uint32_t push_constant_size, uint32_t param_count,
        const spec_fn_t& spec_fn,
        bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_size = 0,
        bool create_aligned = true, bool pin_subgroup_to_warp = false
    ) {
        auto& vec = device->pipeline_matmul[key];
        const bool first_call = vec.empty();
        for (size_t i = 0; i < tile_configs.size(); i++) {
            const auto& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            const uint32_t rsgs = pin_subgroup_to_warp ? tc.warptile[WARP_SIZE_IDX] : required_subgroup_size;
            const bool     rfs  = require_full_subgroups || pin_subgroup_to_warp;

            if (first_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                if (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.empty() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_data, "main", param_count, push_constant_size,
                tc.wg_denoms, spec_fn(tc.warptile, false), 1,
                disable_robustness, rfs, rsgs);

            if (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.empty() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_data, "main", param_count, push_constant_size,
                    tc.wg_denoms, spec_fn(tc.warptile, true), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    auto filter_tc = [&](const std::vector<vk_tile_config>& configs, ggml_type type, bool is_id, bool is_int = false) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> result;
        bool enabled[3];
        if (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } else {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        for (size_t i = 0; i < configs.size() && i < 3; i++) {
            if (enabled[i]) result.push_back(configs[i]);
        }
        return result;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) {
        auto const &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, bool aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_size);     // subgroup_size
            if (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_size((ggml_type)type)); // MmABlockBytes
            }
            return spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, false, true},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_data, sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_data,        sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true);
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_data, sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true);
        }
#endif
        for (const auto type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                continue;
            }
            auto& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, false, true},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_data, sizeof(vk_mat_mat_push_constants), 3, qs, true);
            create_mm_pipelines({type, GGML_TYPE_F16, false, false}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_data,        sizeof(vk_mat_mat_push_constants), 3, qs, true);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, false, true},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_data, sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, false, false}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_data,        sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, false, true},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_data, sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, false, false}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_data,        sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true);
#define X_CM2(TYPE, tstr) \
        { auto tc = filter_tc(tc_mmq, TYPE, false); \
          if (!tc.empty()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, false, true},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, false, false}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        if (device->ocp_fp4) {
#define X_CM2_OCP(TYPE, tstr) \
            { auto tc = filter_tc(tc_mmq, TYPE, false); \
              if (!tc.empty()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, false, true},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, false, false}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        sizeof(vk_mat_mat_push_constants), 3, cm2_spec, true); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#undef X_CM2_OCP
        } else
#endif
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#undef X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, true, true},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_data, sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_data,        sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true);
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_data, sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true);
        }
#endif
        for (const auto type : non_lut_quant_types) {
            if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                continue;
            }
            spec_fn_t qs_id = [&, type](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, true, true},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_data, sizeof(vk_mat_mat_id_push_constants), 5, qs_id, true);
            create_mm_pipelines({type, GGML_TYPE_F16, true, false}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_data,        sizeof(vk_mat_mat_id_push_constants), 5, qs_id, true);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, true, true},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_data, sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, true, false}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_data,        sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, true, true},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_data, sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, true, false}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_data,        sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true);
#define X_CM2_ID(TYPE, tstr) \
        { auto tc = filter_tc(tc_mmqid, TYPE, true); \
          if (!tc.empty()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, true, true},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, true, false}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        if (device->ocp_fp4) {
#define X_CM2_ID_OCP(TYPE, tstr) \
            { auto tc = filter_tc(tc_mmqid, TYPE, true); \
              if (!tc.empty()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, true, true},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, true, false}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        sizeof(vk_mat_mat_id_push_constants), 5, cm2_spec, true); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#undef X_CM2_ID_OCP
        } else
#endif
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#undef X_CM2_ID
    } else
#endif  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        const bool cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        if (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        auto cm1_create = [&](vk_matmul_pipeline_key key, const std::vector<vk_tile_config>& tc_base,
                              const std::string& name, size_t len, const void* data, uint32_t pc_size, uint32_t pc) {
            auto tc = filter_tc(tc_base, key.type_a, key.mul_mat_id);
            if (!tc.empty()) create_mm_pipelines(key, tc, name, len, data, pc_size, pc, cm1_spec, false, true, 0, true, cm1_pin);
        };
        auto cm1_create_quant = [&](vk_matmul_pipeline_key key, const std::vector<vk_tile_config>& tc_base,
                                    const std::string& name, size_t len, const void* data, uint32_t pc_size, uint32_t pc) {
            spec_fn_t qs = [&, type_a=key.type_a](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            auto tc = filter_tc(tc_base, key.type_a, key.mul_mat_id);
            if (!tc.empty()) create_mm_pipelines(key, tc, name, len, data, pc_size, pc, qs, false, true, 0, true, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        auto cm1_create_mmq = [&](vk_matmul_pipeline_key key, const std::vector<vk_tile_config>& tc_base,
                                  const std::string& name, size_t len, const void* data, uint32_t pc_size, uint32_t pc) {
            spec_fn_t identity = [](const std::vector<uint32_t>& wt, bool) { return wt; };
            auto tc = filter_tc(tc_base, key.type_a, key.mul_mat_id, true);
            if (!tc.empty()) create_mm_pipelines(key, tc, name, len, data, pc_size, pc, identity, false, false, 0, false, true);
        };

        std::vector<vk_tile_config> tc_mmq_cm1_int = {
            {s_warptile_mmq_cm1_int, s_mmq_wg_denoms, s_align},
            {m_warptile_mmq_cm1_int, m_mmq_wg_denoms, m_align},
            {l_warptile_mmq_cm1_int, l_mmq_wg_denoms, l_align},
        };
        std::vector<vk_tile_config> tc_mmq_cm1_int_k = {
            {s_warptile_mmq_cm1_int_k, s_mmq_cm1_wg_denoms_k, s_align},
            {m_warptile_mmq_cm1_int_k, m_mmq_cm1_wg_denoms_k, m_align},
            {l_warptile_mmq_cm1_int_k, l_mmq_cm1_wg_denoms_k, l_align},
        };

        // Some quants are not performant on RDNA4, those fall back to FP16 matmul
        const bool rdna3 = device->architecture == AMD_RDNA3;
        const bool rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_data,     sizeof(vk_mat_mat_push_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_data,     sizeof(vk_mat_mat_push_constants), 3);
        if (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, false, true},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, false, true},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
        }
        if (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_data,      sizeof(vk_mat_mat_push_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_data,  sizeof(vk_mat_mat_push_constants), 3);
        }
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
        }
#endif
        for (const auto type : non_lut_quant_types) {
            if (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, false, true},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
            }
            if (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_data,        sizeof(vk_mat_mat_push_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, false, false}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_data,        sizeof(vk_mat_mat_push_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#define X_CM1(TYPE, tstr) \
        if (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, sizeof(vk_mat_mat_push_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, false, true},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, sizeof(vk_mat_mat_push_constants), 3); \
        } \
        if (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        sizeof(vk_mat_mat_push_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, false, false}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        sizeof(vk_mat_mat_push_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        if (device->ocp_fp4) {
#define X_CM1_OCP(TYPE, tstr) \
            if (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, sizeof(vk_mat_mat_push_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, false, true},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, sizeof(vk_mat_mat_push_constants), 3); \
            } \
            if (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        sizeof(vk_mat_mat_push_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, false, false}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        sizeof(vk_mat_mat_push_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#undef X_CM1_OCP
        } else
#endif
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#undef X_CM1

        if (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
            if (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
            if (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_data, sizeof(vk_mat_mat_push_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_data,  sizeof(vk_mat_mat_push_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
            if (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3); }
            if (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_data,   sizeof(vk_mat_mat_push_constants), 3);
            if (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, false, false}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_data,  sizeof(vk_mat_mat_push_constants), 3); }
        }

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        // ggmlR: our subgroup-shuffle MMQ for k-quants (mul_mmq.comp, no shmem
        // round-trip for block_a) on coopmat1 devices too -- it was written for
        // wave64 RDNA, which takes this branch. Registered under key.sg_shuffle
        // next to the cm1 MMQ above; matmul.cpp picks it for wave64 and
        // ne11 >= 512 and falls back to the cm1 MMQ otherwise (9.5-bench).
        if (device->integer_dot_product && device->fp16 && device->subgroup_size == 64) {
            std::vector<vk_tile_config> tc_sg = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
            auto const & sg_shuffle_create = [&](ggml_type type, const char * name, size_t len, const void * data) {
                vk_matmul_pipeline_key key { type, GGML_TYPE_Q8_1, false, false };
                key.sg_shuffle = true;
                auto tc = filter_tc(tc_sg, type, false, true);
                if (!tc.empty()) {
                    spec_fn_t identity = [](const std::vector<uint32_t>& wt, bool) { return wt; };
                    create_mm_pipelines(key, tc, name, len, data, sizeof(vk_mat_mat_push_constants), 3, identity, false, false, 0, false);
                }
            };
            sg_shuffle_create(GGML_TYPE_Q4_K, "matmul_q4_k_q8_1_no_shmem", matmul_q4_k_q8_1_no_shmem_len, matmul_q4_k_q8_1_no_shmem_data);
            sg_shuffle_create(GGML_TYPE_Q5_K, "matmul_q5_k_q8_1_no_shmem", matmul_q5_k_q8_1_no_shmem_len, matmul_q5_k_q8_1_no_shmem_data);
            sg_shuffle_create(GGML_TYPE_Q6_K, "matmul_q6_k_q8_1_no_shmem", matmul_q6_k_q8_1_no_shmem_len, matmul_q6_k_q8_1_no_shmem_data);
        }
#endif

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
        if (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, true, true},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_data,     sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, true, true},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
        }
        if (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_data,     sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
        }
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
        }
#endif
        for (const auto type : non_lut_quant_types) {
            if (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, true, true},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                cm1_create_quant({type, GGML_TYPE_F16, true, true},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            }
            if (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_data,        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                cm1_create_quant({type, GGML_TYPE_F16, true, false}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_data,        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#define X_CM1_ID(TYPE, tstr) \
        if (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
            cm1_create({TYPE, GGML_TYPE_F16, true, true},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
        } \
        if (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
            cm1_create({TYPE, GGML_TYPE_F16, true, false}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        if (device->ocp_fp4) {
#define X_CM1_ID_OCP(TYPE, tstr) \
            if (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
                cm1_create({TYPE, GGML_TYPE_F16, true, true},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
            } \
            if (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
                cm1_create({TYPE, GGML_TYPE_F16, true, false}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#undef X_CM1_ID_OCP
        } else
#endif
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#undef X_CM1_ID

        if (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_data,  sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_data,   sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
            if (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, true, false}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_data,  sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); }
        }
    } else
#endif  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        auto sg_create = [&](vk_matmul_pipeline_key key, const std::vector<vk_tile_config>& tc_base,
                             const std::string& name, size_t len, const void* data, uint32_t pc_size, uint32_t pc,
                             uint32_t rsgs = 0) {
            auto tc = filter_tc(tc_base, key.type_a, key.mul_mat_id);
            if (!tc.empty()) create_mm_pipelines(key, tc, name, len, data, pc_size, pc,
                [&](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_spec(wt, a); },
                false, rsgs > 0, rsgs);
        };
        auto sg_create_quant = [&](vk_matmul_pipeline_key key, const std::vector<vk_tile_config>& tc_base,
                                   const std::string& name, size_t len, const void* data, uint32_t pc_size, uint32_t pc,
                                   uint32_t rsgs = 0) {
            auto tc = filter_tc(tc_base, key.type_a, key.mul_mat_id);
            if (!tc.empty()) {
                spec_fn_t qs = [&, type_a=key.type_a](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(key, tc, name, len, data, pc_size, pc, qs, false, rsgs > 0, rsgs);
            }
        };
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        auto sg_create_mmq = [&](vk_matmul_pipeline_key key, const std::vector<vk_tile_config>& tc_base,
                                 const std::string& name, size_t len, const void* data, uint32_t pc_size, uint32_t pc,
                                 uint32_t rsgs = 0) {
            auto tc = filter_tc(tc_base, key.type_a, key.mul_mat_id, true);
            if (!tc.empty()) {
                spec_fn_t identity = [](const std::vector<uint32_t>& wt, bool) { return wt; };
                create_mm_pipelines(key, tc, name, len, data, pc_size, pc, identity, false, rsgs > 0, rsgs, false);
            }
        };
#endif

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        if (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #define SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## _dot2_data : NAME ## _data)
            #define SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## _dot2_f16acc_data : NAME ## _f16acc_data)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, false, true},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, false, true},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        sizeof(vk_mat_mat_push_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_data, sizeof(vk_mat_mat_push_constants), 3);

            for (const auto type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), sizeof(vk_mat_mat_push_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        sizeof(vk_mat_mat_push_constants), 3);
            }
    #define X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), sizeof(vk_mat_mat_push_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        sizeof(vk_mat_mat_push_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#undef X_SG

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_data, sizeof(vk_mat_mat_push_constants), 3);
                // ggmlR: subgroup-shuffle MMQ for k-quants (no shmem round-trip for
                // block_a), wave64 only; picked via key.sg_shuffle in matmul.cpp.
                if (device->subgroup_size == 64) {
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, false, false, false, true}, tc_mmq_int_k, "matmul_q4_k_q8_1_no_shmem", matmul_q4_k_q8_1_no_shmem_len, matmul_q4_k_q8_1_no_shmem_data, sizeof(vk_mat_mat_push_constants), 3);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, false, false, false, true}, tc_mmq_int_k, "matmul_q5_k_q8_1_no_shmem", matmul_q5_k_q8_1_no_shmem_len, matmul_q5_k_q8_1_no_shmem_data, sizeof(vk_mat_mat_push_constants), 3);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, false, false, false, true}, tc_mmq_int_k, "matmul_q6_k_q8_1_no_shmem", matmul_q6_k_q8_1_no_shmem_len, matmul_q6_k_q8_1_no_shmem_data, sizeof(vk_mat_mat_push_constants), 3);
                }
            }
#endif

            if (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, true},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, true},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                for (const auto type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                }
        #define X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size); \
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#undef X_SG_ID_SUB
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                if (device->integer_dot_product) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                }
#endif
            } else {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, true},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, true},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                for (const auto type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                }
        #define X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#undef X_SG_ID
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                if (device->integer_dot_product) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, true, false}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                }
#endif
            }
            #undef SPV_DOT2
            #undef SPV_DOT2_F16ACC
        } else {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, false, false}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_data,     sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, false, false}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_data,    sizeof(vk_mat_mat_push_constants), 3);

            for (const auto type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
            }
    #define X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#undef X_SG_FP32

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, false, false}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, false, false}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_data, sizeof(vk_mat_mat_push_constants), 3);
            }
#endif

            if (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_data,     sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_data,    sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size_16);
                for (const auto type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                }
        #define X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, mul_mat_subgroup_size);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#undef X_SG_ID_SUB_FP32
            } else {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, true, false}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_data,     sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, true, false}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_data,    sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                for (const auto type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                }
        #define X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#undef X_SG_ID_FP32
            }
        }
    }
#undef FOR_EACH_LUT_TYPE
#undef FOR_EACH_LUT_TYPE_NONFP4
#undef FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    if ((device->coopmat2 || device->coopmat_support)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#endif
        ) {
        const uint32_t s_warptile_wm_bf16 = device->subgroup_size == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_size_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_size_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        auto tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, false);
        auto tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, true);
        spec_fn_t bf16_spec = [&](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_spec(wt, a); };
        if (!tc_bf16_filtered.empty()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, false, false}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_data, sizeof(vk_mat_mat_push_constants), 3, bf16_spec);
        }
        if (!tc_bf16_id_filtered.empty()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, true, false}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_data, sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count, bf16_spec);
        }
    }

    // ggmlR: exact f32 x f32 for nodes with GGML_PREC_F32 (not upstream).
    //
    // On an fp16 or coopmat device every matmul_f32_f32 variant above keeps its
    // shared-memory tiles in f16, so f32 x f32 with n > 8 (the mat-vec path ends
    // at 8) carries ~4e-4 relative error. The _fp32 SPIR-V is always generated;
    // here it is registered under key.prec32 with SCALAR tiles (the coopmat
    // layouts would not match a non-coopmat shader) and only in the sizes whose
    // f32 tiles fit in shared memory. On a plain fp32 device nothing is
    // registered: the pipeline-map lookup falls back to the ordinary f32 entry,
    // which is already exact. Pipelines compile lazily, so callers that never
    // ask for GGML_PREC_F32 (llamaR, sd2R) pay nothing.
    if (device->fp16 || device->coopmat_support || device->coopmat2) {
        const std::vector<uint32_t> p32_l = { 128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 };
        const std::vector<uint32_t> p32_m = { 128,  64,  64, 16, subgroup_size_8,     32, 2, 4, 2, 1, subgroup_size_8 };
        const std::vector<uint32_t> p32_s = { subgroup_size_16, 32, 32, 16, 32,        32, 2, 2, 2, 1, subgroup_size_8 };
        std::vector<vk_tile_config> tc_p32;
        for (const vk_tile_config & tc : std::vector<vk_tile_config>{ { p32_s, { 32, 32, 1 }, 32 }, { p32_m, { 64, 64, 1 }, 64 }, { p32_l, { 128, 128, 1 }, 128 } }) {
            if (ggml_vk_matmul_shmem_support(device, tc.warptile, false, GGML_TYPE_F32, true)) {
                tc_p32.push_back(tc);
            }
        }
        if (!tc_p32.empty()) {
            vk_matmul_pipeline_key key { GGML_TYPE_F32, GGML_TYPE_F32, false, false };
            key.prec32 = true;
            create_mm_pipelines(key, tc_p32, "matmul_f32_f32_prec32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data,
                sizeof(vk_mat_mat_push_constants), 3,
                [&](const std::vector<uint32_t>& wt, bool a) { return ggml_vk_mul_mm_spec(wt, a); });
        }
    }

    // Set up tile selector functions
    if (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_count,
                                          const std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            if (configs.size() <= 1) return 0;
            uint32_t last = (uint32_t)configs.size() - 1;
            if (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                return (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            const uint32_t tiles_l = CEIL_DIV(m, configs[last].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[last].unaligned->wg_denoms[1]);
            const uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            bool prefer_large = tiles_m > shader_core_count || tiles_l > shader_core_count ||
                                (tiles_l <= shader_core_count / 3 && tiles_m > shader_core_count / 2);
            if (n > crossover_large && prefer_large) return last;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            if (m > crossover_medium_m && n > crossover_medium_n) return 1;
            return 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             const std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            if (configs.size() <= 1) return 0;
            uint32_t last = (uint32_t)configs.size() - 1;
            if (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                return (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            if (n > crossover_large) return last;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            if (n > crossover_medium) return 1;
            return 0;
        };
    } else {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          const std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            if (configs.size() <= 1) return 0;
            if (m <= 32 || n <= 32) return 0;
            if (configs.size() == 2) return 1;
            if (m <= 64 || n <= 64) return 1;
            return (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    auto const &rm_iq_int = [](uint32_t i) { return i == 0 ? 8u : 4u; };
    if (device->vendor_id == VK_VENDOR_ID_AMD) {
        if (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } else if (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    const bool is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    auto const &rm_int_n = [&](uint32_t rows, uint32_t i) { return (is_rdna3_or_4 && i >= 4) ? 4u : rows; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    auto const &rm_id = [&](uint32_t rows) {
        if (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            return 4u;
        }
        return is_rdna3_or_4 ? 4u : rows;
    };
    uint32_t rm_iq = 2 * rm_kq;

    const bool use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    const bool is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    const bool use_subgroups16 = use_subgroups && subgroup_min_size_16 && !is_imagination_proprietary;

    const uint32_t subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control && device->subgroup_min_size <= 16 && device->subgroup_max_size >= 16) ? 16 : device->subgroup_size;
    const uint32_t subgroup_size16 = std::max(subgroup_size, 16u);

    const uint32_t force_subgroup_size = use_subgroups ? subgroup_size : 0;
    const uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    static constexpr uint32_t mul_mat_vec_num_bindings = 5;
    static constexpr uint32_t mul_mat_vec_id_num_bindings = 6;

#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#define OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#define OCP_DMMV_DATA(NAME, REDUC) (device->ocp_fp4 ? NAME ## _ocp_data[REDUC] : NAME ## _data[REDUC])
#else
#define OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#define OCP_DMMV_DATA(NAME, REDUC) NAME ## _data[REDUC]
#endif

    for (uint32_t w = 0; w < DMMV_WG_SIZE_COUNT; ++w) {
        const uint32_t wg_size_subgroup   = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size : (subgroup_size * 4);
        const uint32_t wg_size_subgroup16 = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        const shader_reduction_mode reduc = (use_subgroups && w == DMMV_WG_SIZE_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_SIZE_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                            SHADER_REDUCTION_MODE_SHMEM;

        const shader_reduction_mode reduc16 = (use_subgroups16 && w == DMMV_WG_SIZE_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_SIZE_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                              SHADER_REDUCTION_MODE_SHMEM;

        for (uint32_t i = 0; i < mul_mat_vec_max_cols; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, {wg_size_subgroup, 1, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, {wg_size_subgroup, 1, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                const uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
                const uint32_t wg_size_subgroup_int = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_iq_int(i), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_iq_int(i), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_size_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, true, use_subgroups, subgroup_size_int);

            }
#endif // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_data[reduc],     "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, {wg_size_subgroup, 1}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_data[reduc],     "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {wg_size_subgroup, 2}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {wg_size_subgroup, 2}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_DATA(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            const uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
            const uint32_t wg_size_subgroup_int = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(2*rm_kq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(2*rm_stdq_int)}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(2*rm_kq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_kq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_kq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_kq_int)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_kq_int)}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_iq_int(0), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(0)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_iq_int(0), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(0)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_size_subgroup_int, rm_id(1*rm_stdq_int)}, 1, true, use_subgroups, subgroup_size_int);
        }
#endif // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#undef OCP_DMMV_DATA
#undef OCP_DMMV_LEN

#if !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_UNUSED(rm_stdq_int);
    GGML_UNUSED(rm_kq_int);
    GGML_UNUSED(is_rdna3_or_4);
    GGML_UNUSED(rm_int_n);
    GGML_UNUSED(rm_id);
    GGML_UNUSED(rm_iq_int);
#endif

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F32 ], "get_rows_f32",  get_rows_f32_len,  get_rows_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F16 ], "get_rows_f16",  get_rows_f16_len,  get_rows_f16_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_BF16], "get_rows_bf16", get_rows_bf16_len, get_rows_bf16_data, "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q1_0], "get_rows_q1_0", get_rows_q1_0_len, get_rows_q1_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_0], "get_rows_q2_0", get_rows_q2_0_len, get_rows_q2_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_0], "get_rows_q4_0", get_rows_q4_0_len, get_rows_q4_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_1], "get_rows_q4_1", get_rows_q4_1_len, get_rows_q4_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_0], "get_rows_q5_0", get_rows_q5_0_len, get_rows_q5_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_1], "get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q8_0], "get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_K], "get_rows_q2_k", get_rows_q2_k_len, get_rows_q2_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q3_K], "get_rows_q3_k", get_rows_q3_k_len, get_rows_q3_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_K], "get_rows_q4_k", get_rows_q4_k_len, get_rows_q4_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_K], "get_rows_q5_k", get_rows_q5_k_len, get_rows_q5_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q6_K], "get_rows_q6_k", get_rows_q6_k_len, get_rows_q6_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_rows_tq1_0_len, get_rows_tq1_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_rows_tq2_0_len, get_rows_tq2_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_rows_iq1_s_len,   get_rows_iq1_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_rows_iq1_m_len,   get_rows_iq1_m_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_rows_iq2_xxs_len, get_rows_iq2_xxs_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_rows_iq2_xs_len,  get_rows_iq2_xs_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_rows_iq2_s_len,   get_rows_iq2_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_rows_iq3_xxs_len, get_rows_iq3_xxs_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_rows_iq3_s_len,   get_rows_iq3_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_rows_iq4_xs_len,  get_rows_iq4_xs_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_rows_iq4_nl_len,  get_rows_iq4_nl_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_rows_mxfp4_len,   get_rows_mxfp4_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_rows_nvfp4_len,   get_rows_nvfp4_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    // 512, not 1024: this one is built from get_rows.comp, whose thread takes
    // ONE element (i00 = gl_GlobalInvocationID.x), while the quantised
    // pipelines above come from get_rows_quant.comp, whose thread takes two
    // (i00 = gl_GlobalInvocationID.x * 2) and so does cover 1024 per group.
    // With 1024 here CEIL_DIV handed out half the workgroups needed and the
    // back of every row past ne00=512 kept whatever the buffer held: measured
    // on SAGEConv, whose Gather is i32 over ne0=1433, where the GPU output
    // disagreed with the CPU in 99.9% of elements on a realistic edge_index.
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_I32],     "get_rows_i32",     get_rows_i32_len,     get_rows_i32_data,     "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_rows_f32_f32_len,  get_rows_f32_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_rows_f16_f32_len,  get_rows_f16_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_rows_bf16_f32_len, get_rows_bf16_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_rows_q1_0_f32_len, get_rows_q1_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_rows_q2_0_f32_len, get_rows_q2_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_rows_q4_0_f32_len, get_rows_q4_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_rows_q4_1_f32_len, get_rows_q4_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_rows_q5_0_f32_len, get_rows_q5_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_rows_q2_k_f32_len, get_rows_q2_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_rows_q3_k_f32_len, get_rows_q3_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_rows_q4_k_f32_len, get_rows_q4_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_rows_q5_k_f32_len, get_rows_q5_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_rows_q6_k_f32_len, get_rows_q6_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_rows_tq1_0_f32_len, get_rows_tq1_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_rows_tq2_0_f32_len, get_rows_tq2_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_rows_iq1_s_f32_len,   get_rows_iq1_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_rows_iq1_m_f32_len,   get_rows_iq1_m_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_rows_iq2_xxs_f32_len, get_rows_iq2_xxs_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_rows_iq2_xs_f32_len,  get_rows_iq2_xs_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_rows_iq2_s_f32_len,   get_rows_iq2_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_rows_iq3_xxs_f32_len, get_rows_iq3_xxs_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_rows_iq3_s_f32_len,   get_rows_iq3_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_rows_iq4_xs_f32_len,  get_rows_iq4_xs_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_rows_iq4_nl_f32_len,  get_rows_iq4_nl_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_rows_mxfp4_f32_len,   get_rows_mxfp4_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_rows_nvfp4_f32_len,   get_rows_nvfp4_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_split_k_reduce, "split_k_reduce", split_k_reduce_len, split_k_reduce_data, "main", 2, 2 * sizeof(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce, "fa_split_k_reduce", fa_split_k_reduce_len, fa_split_k_reduce_data, "main", 3, sizeof(vk_op_flash_attn_split_k_reduce_push_constants), {1, device->subgroup_size, 1}, {device->subgroup_size}, 1, true);

#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        auto upper_power_of_2 = [&](uint32_t in) {
            GGML_ASSERT(in != 0);
            if (in <= 1) return 1u;
            uint32_t ret = in - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            return ret + 1;
            };

        uint32_t xe_native_sub_group_size = 16;
        if (device->architecture == INTEL_XE1) {
            xe_native_sub_group_size = 8;
        }

        for (auto& it : device->pipeline_xe_fa_decode_dual_phases) {
            const uint32_t split_p_chunk = 32;
            auto HdQk = it.first;
            auto& pipelines = it.second;
            uint32_t head_dim_qk = std::get<0>(HdQk);
            uint32_t head_dim_pv = std::get<1>(HdQk);
            uint32_t gqa_ratio = std::get<2>(HdQk);
            uint32_t q_len = std::get<3>(HdQk);
            const uint32_t out_dim_per_wg = gqa_ratio > 16 ? 8 : 16;
            uint32_t aligned_q_len = upper_power_of_2(q_len);
            uint32_t group_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_group_size, 64u), 256u);
            uint32_t out_per_wg_ph1 = std::min(q_len, 256u / xe_native_sub_group_size);
            uint32_t aligned_gqa_ratio = upper_power_of_2(gqa_ratio);
            uint32_t split_p_per_iter_ph2 = 256;
            uint32_t split_p_per_warp = 16;
            uint32_t group_sz_ph2 = (split_p_per_iter_ph2 / split_p_per_warp) * xe_native_sub_group_size;
            uint32_t out_per_wg_ph2 = std::min(std::max(16u / aligned_gqa_ratio, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.first, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_data, "main", 5, sizeof(vk_fa_xe_opt_push_constants), { 1, 32, 1 }, { group_sz_ph1, gqa_ratio, head_dim_qk, xe_native_sub_group_size, split_p_chunk, out_per_wg_ph1 }, 1, false, true, xe_native_sub_group_size);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_data, "main", 5, sizeof(vk_fa_xe_opt_push_constants), { 1, 1, 1 }, { group_sz_ph2, gqa_ratio, head_dim_pv, out_per_wg_ph2, xe_native_sub_group_size, split_p_per_iter_ph2, split_p_chunk, out_dim_per_wg }, 1, false, true, xe_native_sub_group_size);
        }
    }
#endif

    for (auto &it : device->pipeline_fa_mask_opt) {
        auto BrBc = it.first;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_mask_opt_len, fa_mask_opt_data, "main", 2, sizeof(vk_op_flash_attn_mask_opt_push_constants), {1, 1, 1}, {128, 128 / device->subgroup_size, BrBc.first, BrBc.second}, 1, true, true, device->subgroup_size);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        const uint32_t compact_max = std::min({1024u, device->properties.limits.maxComputeWorkGroupInvocations, device->properties.limits.maxComputeWorkGroupSize[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_full_support;
        if (device->fa_sparse_compact_use_subgroups) {
            const uint32_t compact_wg = std::max(device->subgroup_size, (compact_max / device->subgroup_size) * device->subgroup_size);
            const uint32_t compact_num_sg = compact_wg / device->subgroup_size;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_data, "main", 2, sizeof(vk_op_flash_attn_sparse_compact_push_constants), {1, 1, 1}, {compact_wg, compact_num_sg}, 1, true, true, device->subgroup_size);
        } else {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_data, "main", 2, sizeof(vk_op_flash_attn_sparse_compact_push_constants), {1, 1, 1}, {compact_max}, 1, true);
        }
    }


    if (device->subgroup_clustered && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_data, "main", 2, sizeof(vk_quantize_q8_1_push_constants), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1, true, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_data, "main", 2, sizeof(vk_quantize_q8_1_push_constants), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1);
    }

    for (uint32_t i = 0; i < p021_max_gqa_ratio; ++i) {
        if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_add_len, mul_mat_vec_p021_f16_f32_subgroup_add_data, "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_p021_push_constants), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, true, true);
        } else {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_data,              "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_p021_push_constants), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, true);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_data, "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_nc_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_group_norm_f32, "group_norm_f32", group_norm_f32_len, group_norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_f32, "rms_norm_mul_add_f32", rms_norm_mul_add_f32_len, rms_norm_mul_add_f32_data, "main", 5, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_add_f32_len, rms_norm_mul_add_f32_data, "main", 5, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1, 1}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_add_partials_f32_len, rms_norm_mul_add_partials_f32_data, "main", 6, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_add_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_add_partials_f32_len, rms_norm_mul_add_partials_f32_data, "main", 6, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1, 1}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_set_rows_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_set_rows_f32_f32_len, rms_norm_set_rows_f32_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_set_rows_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_set_rows_f32_f16_len, rms_norm_set_rows_f32_f16_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1}, 1, true);

    if (sizeof(vk_op_rms_norm_mul_rope_push_constants) <= device->properties.limits.maxPushConstantsSize) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_data, "main", 7, sizeof(vk_op_rms_norm_mul_rope_push_constants), {1, 1, 1}, {0, 1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_data, "main", 7, sizeof(vk_op_rms_norm_mul_rope_push_constants), {1, 1, 1}, {0, 1}, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_back_f32, "rms_norm_back_f32", rms_norm_back_f32_len, rms_norm_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    // DIVERGENCE from upstream: LayerNorm backward, see ggml_norm_back().
    ggml_vk_create_pipeline(device, device->pipeline_norm_back_f32, "norm_back_f32", norm_back_f32_len, norm_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);

#define SET_ROWS(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  set_rows_ ## src ## _f32 ## itype ## _len,  set_rows_ ## src ## _f32 ## itype ## _data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  set_rows_ ## src ## _f16 ## itype ## _len,  set_rows_ ## src ## _f16 ## itype ## _data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, set_rows_ ## src ## _bf16 ## itype ## _len, set_rows_ ## src ## _bf16 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, set_rows_ ## src ## _q1_0 ## itype ## _len, set_rows_ ## src ## _q1_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, set_rows_ ## src ## _q2_0 ## itype ## _len, set_rows_ ## src ## _q2_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, set_rows_ ## src ## _q4_0 ## itype ## _len, set_rows_ ## src ## _q4_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, set_rows_ ## src ## _q4_1 ## itype ## _len, set_rows_ ## src ## _q4_1 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, set_rows_ ## src ## _q5_0 ## itype ## _len, set_rows_ ## src ## _q5_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, set_rows_ ## src ## _q5_1 ## itype ## _len, set_rows_ ## src ## _q5_1 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, set_rows_ ## src ## _q8_0 ## itype ## _len, set_rows_ ## src ## _q8_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, set_rows_ ## src ## _iq4_nl ## itype ## _len, set_rows_ ## src ## _iq4_nl ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);

    SET_ROWS(0, f32, _i32)
    SET_ROWS(0, f32, _i64)
    SET_ROWS(1, f16, _i32)
    SET_ROWS(1, f16, _i64)
#undef SET_ROWS


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    auto get_suffix = [](bool src0_f16, bool src1_f16, bool dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        return s;
    };

#define CREATE_BINARY(name, namemod, spec, bindings) \
    for (int s0 : {0,1}) for (int s1 : {0,1}) for (int d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## _data[s0][s1][d], \
                                "main", (bindings), sizeof(vk_op_binary_push_constants), {512, 1, 1}, spec, 1);

    CREATE_BINARY(add, , {0}, 4)
    CREATE_BINARY(add, _norepeat, {1}, 4)
    CREATE_BINARY(sub, , {0}, 3)
    CREATE_BINARY(sub, _norepeat, {1}, 3)
    CREATE_BINARY(mul, , {0}, 3)
    CREATE_BINARY(mul, _norepeat, {1}, 3)
    CREATE_BINARY(div, , {0}, 3)
    CREATE_BINARY(div, _norepeat, {1}, 3)
    CREATE_BINARY(add_rms, , {0}, 4)
    CREATE_BINARY(add_rms, _norepeat, {1}, 4)
#undef CREATE_BINARY

    if (device->multi_add) {
        for (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add[i],     "multi_add_f32_"     + std::to_string(i+1), multi_add_f32_len,     multi_add_f32_data,     "main", MAX_PARAMETER_COUNT, sizeof(vk_op_multi_add_push_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_add_rms_f32_len, multi_add_rms_f32_data, "main", MAX_PARAMETER_COUNT, sizeof(vk_op_multi_add_push_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_add_id_f32, "add_id_f32", add_id_f32_len, add_id_f32_data, "main", 4, sizeof(vk_op_add_id_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_set_f32, "set_f32", acc_f32_len, acc_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {0, 0}, 1);

    /* ⚠️ vk_op_binary_push_constants, NOT vk_op_scatter_elements_push_constants.
     *
     * The push constant range declared here must match what the dispatcher
     * actually sends: ggml_vk_scatter_elements() calls ggml_vk_op_f32 with
     * vk_op_binary_push_constants_init(...), i.e. 140 bytes. This line used to
     * declare the 3-uint (12 byte) struct left over from an earlier version of
     * the shader, so everything the shader reads past offset 12 -- ne02, ne03
     * (the tail of `total`), ne20..ne22 (the dst strides), misalign_offsets and
     * param3 (the axis) -- lay outside the declared range.
     *
     * Measured on MaskRCNN-12-int8 (node 1211, 11226880 updates into a
     * [7,7,256,1000] dst): the output came back with 2 non-zero rows where
     * ONNX Runtime has 805, and 8838153 of 12544000 elements disagreed, while
     * BOTH inputs matched ORT exactly (updates 1.4e-05, indices 0). The CPU
     * kernel matched ORT to 7.6e-06. */
    /* ⚠️ wg_denoms {1,1,1}, not {256,1,1}.
     *
     * ggml_vk_dispatch_pipeline divides again: workgroups = CEIL_DIV(elements,
     * wg_denoms). The elements this op passes are ALREADY workgroup counts --
     * ggml-vulkan-elemwise.cpp computes CEIL_DIV(nelements(updates), 256)
     * before splitting it over the axes -- so a denominator of 256 divided a
     * second time and launched a fraction of the work.
     *
     * Measured: updates of 3072 elements dispatched elements={12,1,1}, which
     * became CEIL_DIV(12,256) = 1 workgroup, 256 threads. The output held
     * exactly the elements written by gid 0..255 -- one destination row of the
     * six expected. At 163840 updates it was 3 rows of 40.
     *
     * GGML_OP_SET_ROWS is the same shape (same push constants, same
     * pre-divided elements) and declares {1,1,1}; keep these in step. */
    ggml_vk_create_pipeline(device, device->pipeline_scatter_elements_none, "scatter_elements_none", scatter_elements_none_len, scatter_elements_none_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_scatter_elements_add,  "scatter_elements_add",  scatter_elements_add_len,  scatter_elements_add_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_nearest_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_NEAREST}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale[0], "scale_f32", scale_f32_len, scale_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_scale[1], "scale_f16", scale_f16_len, scale_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sqr[0], "sqr_f32", sqr_f32_len, sqr_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sqr[1], "sqr_f16", sqr_f16_len, sqr_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sqrt[0], "sqrt_f32", sqrt_f32_len, sqrt_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sqrt[1], "sqrt_f16", sqrt_f16_len, sqrt_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sin[0], "sin_f32", sin_f32_len, sin_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sin[1], "sin_f16", sin_f16_len, sin_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cos[0], "cos_f32", cos_f32_len, cos_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cos[1], "cos_f16", cos_f16_len, cos_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_clamp[0], "clamp_f32", clamp_f32_len, clamp_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_clamp[1], "clamp_f16", clamp_f16_len, clamp_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_data, "main", 2, sizeof(vk_op_pad_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_back_f32, "repeat_back_f32", repeat_back_f32_len, repeat_back_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

#define CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_quick)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(elu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(abs)
    CREATE_UNARY(softplus)
    CREATE_UNARY(step)
    CREATE_UNARY(round)
    CREATE_UNARY(round_even)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
#undef CREATE_UNARY

#define CREATE_UNARY_RTE(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    CREATE_UNARY_RTE(exp)
    CREATE_UNARY_RTE(expm1)
#undef CREATE_UNARY_RTE

// spec constants: {norepeat, op_on_b}
#define CREATE_UNARY_MUL(name, idx) \
    for (int dt = 0; dt < 2; ++dt) { \
        const size_t len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        const unsigned char * data_ = dt ? name ## _mul_f16_data : name ## _mul_f32_data; \
        const std::string dts_ = dt ? "f16" : "f32"; \
        for (int ob = 0; ob < 2; ++ob) \
            for (int nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, data_, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#undef CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_data, "main", 1, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rel_pos_bias_f32, "rel_pos_bias_f32", rel_pos_bias_f32_len, rel_pos_bias_f32_data, "main", 3, sizeof(vk_op_rel_pos_bias_push_constants), {8, 8, 4}, {}, 1);

    // UMAP SGD layout step: 3 buffers (coords rw, edges, weights), one thread
    // per edge. wg_denoms.x = local_size_x in umap_sgd.comp (256).
    ggml_vk_create_pipeline(device, device->pipeline_umap_sgd, "umap_sgd", umap_sgd_len, umap_sgd_data, "main", 3, sizeof(vk_op_umap_sgd_push_constants), {256, 1, 1}, {}, 1);

    // Pairwise squared-distance matrix: 2 buffers (X in, D2 out), one thread per
    // output cell, tiled 32x32 with shared-memory staging. wg_denoms = local_size
    // (32, 32) = the tile side TS in pairwise_dist.comp.
    ggml_vk_create_pipeline(device, device->pipeline_pairwise_dist, "pairwise_dist", pairwise_dist_len, pairwise_dist_data, "main", 2, sizeof(vk_op_pairwise_dist_push_constants), {32, 32, 1}, {}, 1);

    // RoiAlign (ONNX): 4 buffers (X, rois, batch_indices in; dst out), one thread
    // per output element over a flat 1-D grid. wg_denoms = local_size (256, 1, 1)
    // from roi_align.comp, so the dispatch's element count rounds up to whole
    // workgroups and the shader's own bounds test drops the tail.
    ggml_vk_create_pipeline(device, device->pipeline_roi_align, "roi_align", roi_align_len, roi_align_data, "main", 4, sizeof(vk_op_roi_align_push_constants), {256, 1, 1}, {}, 1);

    // QLinearMatMul with an i32 accumulator (ONNX): 5 buffers (A, B, b_scale,
    // b_zp in; dst out), one thread per output element. The K loop stays inside
    // the thread on purpose -- the VPMADDUBSW pair saturation it reproduces is
    // order-dependent, so it cannot be split across a workgroup.
    // wg_denoms {16,16,1} match the shader's 2D block tile (BM x BN): the
    // dispatch is now sized in output COLUMNS by ROWS, not in flat elements.
    ggml_vk_create_pipeline(device, device->pipeline_qmatmul_i32, "qmatmul_i32", qmatmul_i32_len, qmatmul_i32_data, "main", 5, sizeof(vk_op_qmatmul_i32_push_constants), {16, 16, 1}, {}, 1);

    // QLinearConv with an i32 accumulator (ONNX): 6 buffers (x, w, w_scale,
    // w_zp, bias in; dst out), one thread per output element. As with the
    // matmul, the reduction stays inside the thread because the VPMADDUBSW
    // pair saturation is order-dependent.
    ggml_vk_create_pipeline(device, device->pipeline_qconv_i32, "qconv_i32", qconv_i32_len, qconv_i32_data, "main", 7, sizeof(vk_op_qconv_i32_push_constants), {256, 1, 1}, {}, 1);

    // NonMaxSuppression (ONNX): 4 buffers (boxes, scores in; per-pair selected
    // indices and counts out). ONE WORKGROUP PER (batch, class) pair, so
    // wg_denoms is {1,1,1} and the dispatch's element count becomes the group
    // count directly -- the 256 threads inside cooperate on filtering and
    // sorting, while the selection itself is serial on one invocation.
    ggml_vk_create_pipeline(device, device->pipeline_nms, "nms", nms_len, nms_data, "main", 4, sizeof(vk_op_nms_push_constants), {1, 1, 1}, {}, 1);

    // Tiled fused k-NN: 3 buffers (X in, KNN_IDX out, KNN_DIST out), ONE workgroup
    // per query row (wg_denoms = {1,1,1} so the dispatch's n elements become n
    // groups), local_size_x = WG threads sweeping candidates. Specialization
    // constants are { WG, K, MAXD } by id: WG = workgroup size (subgroup-friendly),
    // K = top-k capacity (>= any requested k), MAXD = max feature dims staged to
    // shared. The requested k arrives per-dispatch in the push constants.
    {
        const uint32_t knn_wg   = std::max<uint32_t>(device->subgroup_size, 1u);
        const uint32_t knn_kcap = 32u;   // top-k capacity; matches best_d[32] in the shader
        const uint32_t knn_maxd = 64u;   // max feature dims staged to shared (q_row[MAXD])
        ggml_vk_create_pipeline(device, device->pipeline_knn_tiled, "knn_tiled", knn_tiled_len, knn_tiled_data, "main", 3, sizeof(vk_op_knn_tiled_push_constants), {1, 1, 1}, { knn_wg, knn_kcap, knn_maxd }, 1);
    }

    // FP64 matmul (PoC, benchmark only): 3 buffers (A, B in, C out), one thread
    // per output element, tiled 16x16 with shared-memory staging. wg_denoms =
    // local_size (16, 16) = the tile side TS in matmul_f64.comp. Only created when
    // the device reports fp64 support (otherwise the double kernel is invalid).
    if (device->physical_device.getFeatures().shaderFloat64) {
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f64, "matmul_f64", matmul_f64_len, matmul_f64_data, "main", 3, sizeof(vk_op_matmul_f64_push_constants), {16, 16, 1}, {}, 1);
    }

    // Sparse LogNormalize: 3 buffers (vals rw, factor, col_of_nnz), one thread
    // per stored non-zero. wg_denoms.x = local_size_x in sparse_lognorm.comp (256).
    ggml_vk_create_pipeline(device, device->pipeline_sparse_lognorm, "sparse_lognorm", sparse_lognorm_len, sparse_lognorm_data, "main", 3, sizeof(vk_op_sparse_lognorm_push_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_data, "main", 1, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_data, "main", 1, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

#define CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_quick)
#undef CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_leaky_relu[0], "leaky_relu_f32", leaky_relu_f32_len, leaky_relu_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_leaky_relu[1], "leaky_relu_f16", leaky_relu_f16_len, leaky_relu_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_silu_back_f32, "silu_back_f32", silu_back_f32_len, silu_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_gelu_back_f32, "gelu_back_f32", gelu_back_f32_len, gelu_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_mask_inf_f32, "diag_mask_inf_f32", diag_mask_inf_f32_len, diag_mask_inf_f32_data, "main", 2, sizeof(vk_op_diag_mask_push_constants), {1, 512, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32,        "soft_max_f32",        soft_max_f32_len,     soft_max_f32_data,     "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg128,  "soft_max_f32_wg128",  soft_max_f32_len,     soft_max_f32_data,     "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512,  "soft_max_f32_wg512",  soft_max_f32_len,     soft_max_f32_data,     "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16,    "soft_max_f32_f16",    soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg128, "soft_max_f32_f16_wg128", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f16,        "soft_max_f16",        soft_max_f16_len,     soft_max_f16_data,     "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f16_wg128,  "soft_max_f16_wg128",  soft_max_f16_len,     soft_max_f16_data,     "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f16_wg512,  "soft_max_f16_wg512",  soft_max_f16_len,     soft_max_f16_data,     "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_back_f32, "soft_max_back_f32", soft_max_back_f32_len, soft_max_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    for (uint32_t i = 0; i < num_argsort_pipelines; ++i) {
        uint32_t BLOCK_SIZE = 1u << std::min(i, device->max_workgroup_size_log2);
        if (i <= device->max_workgroup_size_log2 &&
            2 * sizeof(int) * BLOCK_SIZE <= device->properties.limits.maxComputeSharedMemorySize) {
            const uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_data, "main", 3, sizeof(vk_op_argsort_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, NCOLS_PADDED_LOG2}, 1, true);
        }
        const uint32_t WG_UNROLL_FACTOR = BLOCK_SIZE > 1 ? 2 : 1;
        BLOCK_SIZE /= WG_UNROLL_FACTOR;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_data, "main", 3, sizeof(vk_op_argsort_push_constants), {BLOCK_SIZE * WG_UNROLL_FACTOR, 1, 1}, {BLOCK_SIZE, WG_UNROLL_FACTOR}, 1, true);
    }

    for (uint32_t i = 0; i < num_topk_pipelines; ++i) {
        const uint32_t BLOCK_SIZE = 1u << i;
        const uint32_t NCOLS_PADDED_LOG2 = i;
        if (i <= device->max_workgroup_size_log2) {
            uint32_t nary_shmem = 2 * sizeof(int) * BLOCK_SIZE +
                                  sizeof(int) * device->subgroup_size +
                                  2 * sizeof(int) +
                                  2 * (BLOCK_SIZE / device->subgroup_size) * sizeof(int);
            if (device->subgroup_arithmetic && device->subgroup_require_full_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.maxComputeSharedMemorySize) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_search_f32_len, topk_nary_search_f32_data, "main", 2, sizeof(vk_op_topk_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, device->subgroup_size, device->subgroup_size_log2}, 1, true, true, device->subgroup_size);
            } else if (2 * sizeof(int) * BLOCK_SIZE <= device->properties.limits.maxComputeSharedMemorySize) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_data, "main", 2, sizeof(vk_op_topk_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, NCOLS_PADDED_LOG2}, 1, true);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        const uint32_t BLOCK_SIZE = 1u << std::min(10u, device->max_workgroup_size_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_data, "main", 5, sizeof(vk_op_topk_radix_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, 0}, 1, true);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_data, "main", 5, sizeof(vk_op_topk_radix_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, 1}, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sum_rows[0], "sum_rows_f32", sum_rows_f32_len, sum_rows_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sum_rows[1], "sum_rows_f16", sum_rows_f16_len, sum_rows_f16_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    const uint32_t cumsum_elem_per_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { 256, device->subgroup_size, cumsum_elem_per_thread }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { 128, device->subgroup_size, 1 }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_data, "main", 3, sizeof(vk_op_sum_rows_push_constants), {256, 1, 1}, { 256, device->subgroup_size }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_data, "main", 3, sizeof(vk_op_sum_rows_push_constants), {256, 1, 1}, { 256, device->subgroup_size }, 1, true, true, device->subgroup_size);

    ggml_vk_create_pipeline(device, device->pipeline_count_equal_i32, "count_equal_i32", count_equal_i32_len, count_equal_i32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, { device->subgroup_size }, 1);

    if (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_count_experts, "count_experts", count_experts_subgroup_len, count_experts_subgroup_data, "main", 2, sizeof(vk_op_count_experts_push_constants), {1, 1, 1}, {}, 1, true, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_count_experts, "count_experts", count_experts_len, count_experts_data, "main", 2, sizeof(vk_op_count_experts_push_constants), {1, 1, 1}, {}, 1, true);
    }

    for (auto &s : device->pipeline_solve_tri_f32) {
        const vk_solve_tri_pipeline_state &state = s.first;

        // Max number of rows to load at a time, limited by shared memory
        const uint32_t batch_N = device->properties.limits.maxComputeSharedMemorySize / ((state.N + state.K) * sizeof(float));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        const uint32_t block_size = std::max(128u, 1u << (uint32_t)ceilf(log2f(float(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_data, "main", 3,
            sizeof(vk_op_binary_push_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, block_size }, 1, true);
    }

#define IM2COL(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);
    if (device->shader_int64 && device->buffer_device_address) {
        IM2COL(_bda)
    } else {
        IM2COL()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_data, "main", 2, sizeof(vk_op_timestep_embedding_push_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_data, "main", 3, sizeof(vk_op_conv_transpose_1d_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_data,  "main", 2, sizeof(vk_op_col2im_1d_push_constants), {256, 1, 1}, {}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_data,  "main", 2, sizeof(vk_op_col2im_1d_push_constants), {256, 1, 1}, {}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_data, "main", 2, sizeof(vk_op_col2im_1d_push_constants), {256, 1, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_data,  "main", 4, sizeof(vk_op_snake_push_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_data,  "main", 4, sizeof(vk_op_snake_push_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_data, "main", 4, sizeof(vk_op_snake_push_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_data, "main", 2, sizeof(vk_op_pool1d_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_data, "main", 2, sizeof(vk_op_pool2d_push_constants), {1, 1, 1}, {512}, 1);
    if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_data, "main", 2, sizeof(vk_op_pool2d_push_constants), {1, 1, 1}, {128}, 1, false, true, device->subgroup_size);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_data, "main", 7, sizeof(vk_op_rwkv_wkv6_push_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_data, "main", 8, sizeof(vk_op_rwkv_wkv7_push_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_linear_attn_f32, "gated_linear_attn_f32", gated_linear_attn_f32_len, gated_linear_attn_f32_data, "main", 6, sizeof(vk_op_gated_linear_attn_push_constants), {1, 1, 1}, {}, 1);

    // ggml 0.27: DeepSeek V4 hyper-connections.

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    if (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_full_support && device->subgroup_size >= 16) {
        const uint32_t tokens_per_workgroup = 4 * (device->subgroup_size / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_data, "main", 4, sizeof(vk_op_dsv4_hc_comb_push_constants), {tokens_per_workgroup, 1, 1}, { device->subgroup_size }, 1, true, true, device->subgroup_size);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_data,  "main", 3, sizeof(vk_op_dsv4_hc_pre_push_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_data,  "main", 3, sizeof(vk_op_dsv4_hc_pre_push_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_data, "main", 5, sizeof(vk_op_dsv4_hc_post_push_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_data, "main", 5, sizeof(vk_op_dsv4_hc_post_push_constants), {256, 1, 1}, { 256, 0 }, 1);

    // ggml 0.27: DSA lightning indexer, one pipeline per supported K type.
    for (ggml_type k_type : lightning_indexer_k_types) {
        const std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_data, "main", 5, sizeof(vk_op_lightning_indexer_push_constants), {1, 1, 1}, {(uint32_t)k_type, fa_block_bytes(k_type)}, 1, true);
    }

    {
        const uint32_t gdn_sizes[] = {32, 64, 128};
        const char * gdn_names[][2] = {
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        const bool use_subgroup_reduce = device->subgroup_arithmetic;
        for (uint32_t si = 0; si < 3; si++) {
            const uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            uint32_t lanes_per_column;
            if (S_V >= 128u && device->subgroup_clustered) {
                lanes_per_column = 8u;
            } else {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_per_column = std::min(S_V, device->subgroup_size);
            }

            const bool need_clustered_shader = lanes_per_column != 1 && (lanes_per_column < device->subgroup_size);
            size_t gdn_len;
            const void * gdn_data;
            if (use_subgroup_reduce && need_clustered_shader) {
                gdn_len = gated_delta_net_f32_len;
                gdn_data = (const void *)gated_delta_net_f32_data;
            } else if (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_data = (const void *)gated_delta_net_f32_nocluster_data;
            } else {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_data = (const void *)gated_delta_net_f32_shmem_data;
            }

            const uint32_t cols_per_wg = device->subgroup_size / lanes_per_column;
            const std::array<uint32_t, 3> wg_denoms = {1u, 1u, cols_per_wg};

            for (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_names[si][kda], gdn_len, gdn_data, "main", 7, sizeof(vk_op_gated_delta_net_push_constants),
                    wg_denoms, {S_V, kda, device->subgroup_size, lanes_per_column}, 1, true, use_subgroup_reduce, device->subgroup_size);
            }
        }
    }

    if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {128, device->subgroup_size}, 1, true, true);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {256, device->subgroup_size}, 1, true, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_f32_len, ssm_scan_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {128, device->subgroup_size, 16}, 1, true, true);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_f32_len, ssm_scan_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {256, device->subgroup_size, 16}, 1, true, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, sizeof(vk_op_ssm_conv_push_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, sizeof(vk_op_ssm_conv_push_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, sizeof(vk_op_ssm_conv_push_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);
    // ggmlR extension: 4 buffers (sx, c, grad, packed dst), one invocation per
    // channel, so a plain 1D workgroup rather than ssm_conv's 2D tiling.
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_back_f32, "ssm_conv_back_f32", ssm_conv_back_f32_len, ssm_conv_back_f32_data, "main", 4, sizeof(vk_op_ssm_conv_back_push_constants), {256, 1, 1}, {256}, 1);
    // ggmlR extension: backward of ssm_scan. 10 buffers (7 forward inputs, the
    // incoming gradient, the packed output, and the replay scratch); d_state is
    // a specialisation constant and also the workgroup width, one invocation
    // per state.
    if (device->atomic_float_add) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_back_f32_d128, "ssm_scan_back_128_f32", ssm_scan_back_f32_len, ssm_scan_back_f32_data, "main", 10, sizeof(vk_op_ssm_scan_back_push_constants), {1, 1, 1}, {128}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_back_f32_d256, "ssm_scan_back_256_f32", ssm_scan_back_f32_len, ssm_scan_back_f32_data, "main", 10, sizeof(vk_op_ssm_scan_back_push_constants), {1, 1, 1}, {256}, 1);
    }
    // wg_denoms = the dst tile of one workgroup (TM x TN in out_prod.comp).
    ggml_vk_create_pipeline(device, device->pipeline_out_prod_f32, "out_prod_f32", out_prod_f32_len, out_prod_f32_data, "main", 3, sizeof(vk_op_out_prod_push_constants), {32, 32, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_back_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_back_f32_len, cross_entropy_loss_back_f32_data, "main", 4, sizeof(vk_op_cross_entropy_loss_back_push_constants), {1, 1, 1}, {32}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    const bool can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_in_range(device->properties.driverVersion, 101, 8509, 101, 8860);
    if (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        int idx = 0;
        for (uint32_t n : {64, 128, 256, 512}) {
            if (device->subgroup_size <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n }, 1, true, true, device->subgroup_size);
            }
            ++idx;
        }
    } else if (can_use_fwht) {
        int idx = 0;
        for (uint32_t n : {64, 128, 256, 512}) {
            const uint32_t block_size = std::min(device->subgroup_size, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n }, 1);
            ++idx;
        }
    }

    // ggmlR: backward of flash attention. Six buffers (q, k, v, mask, d, and the
    // packed output). dk/dv accumulate with atomics across query rows, so this
    // needs shaderBufferFloat32AtomicAdd like ssm_scan_back does. The second
    // specialisation constant caps the kv length the shared-memory cache holds;
    // supports_op refuses anything longer rather than overrunning it.
    if (device->atomic_float_add) {
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_back_f32, "flash_attn_back_f32", flash_attn_back_f32_len, flash_attn_back_f32_data, "main", 6, sizeof(vk_op_flash_attn_back_push_constants), {1, 1, 1}, {128, 1024}, 1);
        // Stage-truncated copies for profiling (GGMLR_FAB_PROFILE). Same SPIR-V,
        // different PROFILE_STAGE specialisation constant.
        for (uint32_t st = 1; st <= 4; st++) {
            const std::string prof_name = "flash_attn_back_f32_prof" + std::to_string(st);
            ggml_vk_create_pipeline2(device, device->pipeline_flash_attn_back_f32_prof[st],
                prof_name, flash_attn_back_f32_len, flash_attn_back_f32_data,
                "main", 6, sizeof(vk_op_flash_attn_back_push_constants),
                {1, 1, 1}, {128, 1024, st}, 1);

        }
    }
    // ggmlR: backward of convolution. Two buffers, no atomics -- each invocation
    // owns one output element and gathers the kernel taps that reached it.
    ggml_vk_create_pipeline(device, device->pipeline_im2col_back_f32, "im2col_back_f32", im2col_back_f32_len, im2col_back_f32_data, "main", 2, sizeof(vk_op_im2col_back_push_constants), {256, 1, 1}, {}, 1);

    // ggmlR: backward of embedding lookup. Needs shaderBufferFloat32AtomicAdd:
    // repeated tokens in a batch scatter into the same table row.
    if (device->atomic_float_add) {
        ggml_vk_create_pipeline(device, device->pipeline_get_rows_back_f32, "get_rows_back_f32", get_rows_back_f32_len, get_rows_back_f32_data, "main", 3, sizeof(vk_op_get_rows_back_push_constants), {256, 1, 1}, {}, 1);
    }


    ggml_vk_create_pipeline(device, device->pipeline_opt_step_adamw_f32, "opt_step_adamw_f32", opt_step_adamw_f32_len, opt_step_adamw_f32_data, "main", 5, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_sgd_f32, "opt_step_sgd_f32", opt_step_sgd_f32_len, opt_step_sgd_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d
    for (uint32_t s = 0; s < CONV_SHAPE_COUNT; ++s) {
        uint32_t conv2d_WG_SIZE  = 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_block_size conv2d_BS = vk_conv_block_sizes[s];
        bool conv2d_UNROLL = true;

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#endif

        if (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = false;
        } else if (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            if (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = false;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        bool allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        bool allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        if (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_size,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        uint32_t conv2d_shmem_req =
            (conv2d_BS.K * (conv2d_BS.CRS + conv2d_SHMEM_PAD) + conv2d_BS.CRS * (conv2d_BS.NPQ + conv2d_SHMEM_PAD)) * sizeof(float);
        if (device->properties.limits.maxComputeSharedMemorySize < conv2d_shmem_req) {
            conv2d_BS.CRS = 8;
            if (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_size, conv2d_BS.CRS);
            }
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_SIZE, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

#define CREATE_CONV(name, type_suffix, spv_suffix) \
        for (auto &c : device->pipeline_##name##type_suffix[s]) { \
            const vk_conv2d_pipeline_state &state = c.first;  \
            std::vector<uint32_t> spec_constants_cpy = spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.aligned); \
            ggml_vk_create_pipeline( \
                device, c.second, #name #type_suffix, \
                name##type_suffix##spv_suffix##_len, name##type_suffix##spv_suffix##_data, "main", 3, \
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants_cpy, 1, true, use_collectives);    \
        }
#define CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } else
#endif
        if (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } else {
            CREATE_CONVS( )
        }

#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        // cm1 pipelines use fixed 64×64 tiles regardless of shape slot.
        // Only compile once, into the CONV_SHAPE_128x128 slot which is used at dispatch.
        if (device->coopmat_support && !device->coopmat2 && s == CONV_SHAPE_128x128) {
            // coopmat KHR (cm1) path: subgroup-scope 16x16x16 tiles.
            // Fixed tile sizes independent of scalar-path conv2d_BS:
            //   BS_K=64, BS_NPQ=64, BS_CRS=16, CM_TM/N/K=16
            //   4 subgroups per WG along K → WG_SIZE = subgroup_size * 4
            // wave64 (RDNA): 64*4=256 threads; wave32 (Turing+): 32*4=128 threads.
            // Larger tiles (128x128) give a 512-thread WG on wave64 which hurts
            // occupancy — 256 threads is the sweet spot for RDNA4.
            const uint32_t cm1_BS_K   = 64;
            const uint32_t cm1_BS_NPQ = 64;
            const uint32_t cm1_BS_CRS = 16;
            const uint32_t cm1_n_subgroups = cm1_BS_K / 16;  // == 4
            const uint32_t cm1_wg_size = device->subgroup_size * cm1_n_subgroups;
            std::vector<uint32_t> cm1_spec = { cm1_wg_size, cm1_BS_K, cm1_BS_CRS, cm1_BS_NPQ,
                                               /*TS_K (unused in cm1)=*/4u, /*use_collectives=*/0u, /*SHMEM_PAD=*/8u };
            std::array<uint32_t, 3> cm1_wg_denoms = { cm1_BS_K, 1, 1 };
#define CREATE_CONV_CM1(name, type_suffix) \
            for (auto &c : device->pipeline_##name##type_suffix##_cm1[s]) { \
                const vk_conv2d_pipeline_state &state = c.first;  \
                std::vector<uint32_t> sc = cm1_spec; \
                sc.push_back(state.s0); \
                sc.push_back(state.s1); \
                sc.push_back(state.p0); \
                sc.push_back(state.p1); \
                sc.push_back(state.d0); \
                sc.push_back(state.d1); \
                sc.push_back(state.KW); \
                sc.push_back(state.KH); \
                ggml_vk_create_pipeline( \
                    device, c.second, #name #type_suffix "_cm1", \
                    name##type_suffix##_cm1_cm1_len, name##type_suffix##_cm1_cm1_data, "main", 3, \
                    sizeof(vk_op_conv2d_push_constants), cm1_wg_denoms, sc, 1, true, false); \
            }
            CREATE_CONV_CM1(conv2d, _f32)
            CREATE_CONV_CM1(conv2d, _f16_f32)
            CREATE_CONV_CM1(conv_transpose_2d, _f32)
            CREATE_CONV_CM1(conv_transpose_2d, _f16_f32)
#undef CREATE_CONV_CM1
        }
#endif  // GGML_VULKAN_COOPMAT_GLSLC_SUPPORT

#undef CREATE_CONV
#undef CREATE_CONVS

        // ggmlR: conv3d (ggml 0.27). Our conv2d above keeps its own tuning and
        // our cm1 variant; conv3d_mm.comp is the 0.27 shader and expects the
        // 0.27 parameter set (aligned/csh_store/WM/WN), so it is derived here
        // exactly as 0.27 derives it for conv2d, in its own scope.
        {
            uint32_t conv2d_WG_SIZE  = (s == CONV_SHAPE_64x32) ? 128 : 256;
            uint32_t use_collectives = 0;
            uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
            uint32_t conv2d_SHMEM_PAD = 4;
            vk_conv_block_size conv2d_BS = vk_conv_block_sizes[s];
            bool conv2d_UNROLL = true;

    #if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            if (device->coopmat2) {
                conv2d_SHMEM_PAD = 8; // 8 float16_t
            }
    #endif

            if (device->vendor_id == VK_VENDOR_ID_INTEL) {
                conv2d_SHMEM_PAD = 0;
                conv2d_UNROLL = false;
            } else if (device->vendor_id == VK_VENDOR_ID_AMD) {
                conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
                if (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                    conv2d_UNROLL = false;
                }
            }

            // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
            bool allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                        device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
            bool allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                         device->architecture == vk_device_architecture::AMD_GCN;

            if (device->subgroup_shuffle &&
                device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
                allow_collectives_nv &&
                allow_collectives_amd) {
                use_collectives = 1;
                conv2d_BS.CRS   = std::min(
                    device->subgroup_size,
                    conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
            }

            // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
            // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
            // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
            // subgroup_size_control to force the driver to actually use it.
            bool conv2d_use_cm1 = false;
    #if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            conv2d_use_cm1 = !device->coopmat2 &&
                             device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                             device->subgroup_size_control &&
                             (device->subgroup_size == 32 || device->subgroup_size == 64) &&
                             s != CONV_SHAPE_128x128;
    #endif

            const uint32_t conv2d_cm1_shmem_pad = 8;

            auto shmem_req = [&](uint32_t pad, bool csh_store, bool fp16_shmem) {
                const uint32_t elem_size = fp16_shmem ? (uint32_t)sizeof(uint16_t) : (uint32_t)sizeof(float);
                const uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
                return (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_size;
            };

            // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
            // layout. cm1 needs Csh for output, so check before applying cm1 params.
            if (conv2d_use_cm1 && device->properties.limits.maxComputeSharedMemorySize < shmem_req(conv2d_cm1_shmem_pad, true, true)) {
                conv2d_use_cm1 = false;
            }

            uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
            if (conv2d_use_cm1) {
                conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
                // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
                // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
                const bool sg64 = (device->subgroup_size == 64);
                switch (s) {
                    case CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; break;
                    case CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; break;
                    case CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; break;
                    default: break;
                }
                const uint32_t warps_M = conv2d_BS.K / conv2d_WM;
                const uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
                conv2d_WG_SIZE         = warps_M * warps_N * device->subgroup_size;
            }

            // stage cm2 accumulator through shmem for coalesced global stores;
            // skipped on 128x128 where the extra Csh footprint hurts occupancy.
            // cm1 always uses the staged path.
            uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
            if (conv2d_use_cm1) {
                conv2d_csh_store = 1;
            }

            // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
            const bool conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

            // shrink CRS if the non-cm1 config still doesn't fit
            if (device->properties.limits.maxComputeSharedMemorySize < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
                GGML_ASSERT(!conv2d_use_cm1);
                conv2d_BS.CRS = 8;
                if (use_collectives) {
                    conv2d_BS.CRS = std::min(device->subgroup_size, conv2d_BS.CRS);
                }
                conv2d_csh_store = 0;
            }

            std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };

            // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
            const uint32_t conv2d_required_subgroup_size = conv2d_use_cm1 ? device->subgroup_size : 0;

            std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_SIZE, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
    #define CREATE_CONV3D(type_suffix, spv_suffix) \
            for (auto &c : device->pipeline_conv3d##type_suffix[s]) { \
                const vk_conv3d_pipeline_state &state = c.first; \
                std::vector<uint32_t> spec_constants_cpy = conv3d_spec_constants; \
                spec_constants_cpy.push_back(state.s0); \
                spec_constants_cpy.push_back(state.s1); \
                spec_constants_cpy.push_back(state.s2); \
                spec_constants_cpy.push_back(state.p0); \
                spec_constants_cpy.push_back(state.p1); \
                spec_constants_cpy.push_back(state.p2); \
                spec_constants_cpy.push_back(state.d0); \
                spec_constants_cpy.push_back(state.d1); \
                spec_constants_cpy.push_back(state.d2); \
                spec_constants_cpy.push_back(state.KW); \
                spec_constants_cpy.push_back(state.KH); \
                spec_constants_cpy.push_back(state.KD); \
                spec_constants_cpy.push_back(state.aligned); \
                spec_constants_cpy.push_back(conv2d_csh_store); \
                spec_constants_cpy.push_back(conv2d_WM); \
                spec_constants_cpy.push_back(conv2d_WN); \
                ggml_vk_create_pipeline( \
                    device, c.second, "conv3d" #type_suffix, \
                    conv3d##type_suffix##spv_suffix##_len, conv3d##type_suffix##spv_suffix##_data, "main", 3, \
                    sizeof(vk_op_conv3d_push_constants), wg_denoms, spec_constants_cpy, 1, true, conv2d_required_subgroup_size != 0, conv2d_required_subgroup_size); \
            }
    #if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            if (device->coopmat2) {
                CREATE_CONV3D(_f32, _cm2)
                CREATE_CONV3D(_f16_f32, _cm2)
            } else
    #endif
    #if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            if (conv2d_use_cm1) {
                CREATE_CONV3D(_f32, _cm1)
                CREATE_CONV3D(_f16_f32, _cm1)
            } else
    #endif
            if (conv2d_UNROLL) {
                CREATE_CONV3D(_f32, _unroll)
                CREATE_CONV3D(_f16_f32, _unroll)
            } else {
                CREATE_CONV3D(_f32, )
                CREATE_CONV3D(_f16_f32, )
            }
    #undef CREATE_CONV3D
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);

    for (uint32_t use_push = 0; use_push < 2; ++use_push) {
        for (uint32_t i = 0; i < num_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_push], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_data, "main", 4, sizeof(vk_op_topk_moe_push_constants), {1, 1, 1}, {device->subgroup_size, 1u<<i, use_push}, 1, true, true, device->subgroup_size);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_lock.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    // ggmlR: compiled synchronously in the calling thread (no std::async),
    // which also avoids the MinGW CRT heap race seen during Flux2 model load.
    if (has_claimed_task) {
        auto & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_size, task.spv_data,
                                     task.entrypoint, task.parameter_count, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_full_subgroups, task.required_subgroup_size);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    if (wait_pipeline) {
        std::unique_lock<std::mutex> wait_lock(device->compile_mutex);
        device->compile_cv.wait(wait_lock, [&] {
            return wait_pipeline->compiled.load();
        });
    }
}

static bool ggml_vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties& props, const vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);

static vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (vk_instance.devices[idx] == nullptr) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();


        size_t dev_num = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        if (dev_num >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_num << " does not exist." << std::endl;
            throw std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_num];
        const std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        const char* GGML_VK_PREFER_HOST_MEMORY = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_memory = GGML_VK_PREFER_HOST_MEMORY != nullptr;

        const char* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != nullptr;

        // ggmlR: decide for ourselves whether host-visible VRAM is real ReBAR.
        //
        // ggml_vk_create_buffer_device() asks first for DEVICE_LOCAL |
        // HOST_VISIBLE ("use rebar if available") and never checks that ReBAR
        // is on. Without it that type is only the legacy BAR window -- 256 MiB
        // on RX 9070 / RADV -- and buffers that do not fit there end up in
        // memory the GPU reaches over PCIe. Measured on that machine:
        // ScatterElements(add) 107 ms -> 2-3 ms, SAGEConv 127 -> 10 ms,
        // MaskRCNN-12-int8 ~910 -> ~820 ms once the window is avoided.
        //
        // Decided from heap SIZES only, never from the driver's name or
        // version: the largest heap behind a DEVICE_LOCAL|HOST_VISIBLE type
        // must hold at least GGML_VK_REBAR_MIN_HEAP_FRACTION of the largest
        // DEVICE_LOCAL heap. A 256 MiB window against 16 GiB of VRAM fails by
        // a wide margin; a resized BAR, which spans the VRAM, passes.
        //
        // Overrides: GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM forces the window off
        // (as before); GGML_VK_FORCE_HOST_VISIBLE_VIDMEM keeps it on regardless.
        {
            constexpr double GGML_VK_REBAR_MIN_HEAP_FRACTION = 0.5;
            const bool force_hv = getenv("GGML_VK_FORCE_HOST_VISIBLE_VIDMEM") != nullptr;
            if (!device->disable_host_visible_vidmem && !force_hv) {
                const vk::PhysicalDeviceMemoryProperties mp =
                    device->physical_device.getMemoryProperties();
                const vk::MemoryPropertyFlags hv_vram =
                    vk::MemoryPropertyFlagBits::eDeviceLocal |
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent;
                vk::DeviceSize max_vram = 0, max_hv = 0;
                for (uint32_t h = 0; h < mp.memoryHeapCount; h++) {
                    if (mp.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                        max_vram = std::max(max_vram, mp.memoryHeaps[h].size);
                    }
                }
                for (uint32_t t = 0; t < mp.memoryTypeCount; t++) {
                    if ((mp.memoryTypes[t].propertyFlags & hv_vram) == hv_vram) {
                        max_hv = std::max(max_hv, mp.memoryHeaps[mp.memoryTypes[t].heapIndex].size);
                    }
                }
                if (max_hv > 0 && max_vram > 0 &&
                    (double)max_hv < GGML_VK_REBAR_MIN_HEAP_FRACTION * (double)max_vram) {
                    device->disable_host_visible_vidmem = true;
                    // Printed only under GGMLR_LOG_DEVICE, like the other device
                    // telemetry: the decision itself is unconditional, but tests
                    // create many contexts and the line would repeat for each.
                    const char * log_dev = getenv("GGMLR_LOG_DEVICE");
                    if (log_dev != nullptr && log_dev[0] != '\0' && log_dev[0] != '0') {
                        GGML_LOG_INFO("ggml_vulkan: no Resizable BAR (host-visible VRAM heap "
                                      "%.0f MiB of %.0f MiB) -- device buffers use device-local "
                                      "memory only; GGML_VK_FORCE_HOST_VISIBLE_VIDMEM=1 overrides\n",
                                      (double)max_hv / 1048576.0, (double)max_vram / 1048576.0);
                    }
                }
            }
        }

        const char* GGML_VK_ALLOW_SYSMEM_FALLBACK = getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != nullptr;

        const char* GGML_VK_DISABLE_GRAPH_OPTIMIZE = getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_graph_optimize = GGML_VK_DISABLE_GRAPH_OPTIMIZE != nullptr;

        bool fp16_storage = false;
        bool fp16_compute = false;
        bool maintenance4_support = false;
        bool sm_builtins = false;
        bool amd_shader_core_properties = false;
        bool amd_shader_core_properties2 = false;
        bool pipeline_robustness = false;
        bool coopmat2_support = false;
        bool coopmat2_decode_vector_support = false;
        bool pipeline_executable_properties_support = false;
        bool push_descriptor_ext = false;
        bool internally_sync_support = false;
        device->coopmat_support = false;
        device->integer_dot_product = false;
        device->shader_64b_indexing = false;
        // ggmlR: presence of the extension only -- the actual
        // shaderBufferFloat32AtomicAdd feature bit is read after
        // vkGetPhysicalDeviceFeatures2 below.
        bool atomic_float_ext = false;
        bool bfloat16_support = false;
        bool dot2_f16_support = false;
        bool ocp_microscaling_extension = false;
        bool shader_float8_extension = false;

        for (const auto& properties : ext_props) {
            if (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = true;
            } else if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = true;
            } else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = true;
            } else if (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = true;
            } else if (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = true;
            } else if (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = true;
            } else if (strcmp("VK_EXT_shader_atomic_float", properties.extensionName) == 0) {
                atomic_float_ext = true;
            } else if (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = true;
            } else if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_size_control = true;
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = true;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#endif
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } else if (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = true;
#endif
            } else if (strcmp(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_vector_support = true;
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_product = true;
#endif
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } else if (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = true;
#endif
#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } else if (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = true;
#endif
#if defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } else if (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = true;
#endif
            } else if (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = true;
            } else if (strcmp("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = true;
            } else if (strcmp("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       getenv("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->memory_priority = true;
            } else if (strcmp("VK_KHR_push_descriptor", properties.extensionName) == 0) {
                push_descriptor_ext = true;
            } else if (strcmp("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_memory_host = true;
            } else if (strcmp("VK_KHR_external_memory_fd", properties.extensionName) == 0) {
                // ggmlR Tensor Parallelism (P2P): opaque-fd export/import.
                device->external_memory_fd = true;
#if defined(VK_EXT_shader_64bit_indexing)
            } else if (strcmp("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = true;
#endif
            } else if (strcmp(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = true;
            } else if (strcmp("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = true;
            }
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceMaintenance3Properties props3;
        vk::PhysicalDeviceMaintenance4Properties props4;
        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        vk::PhysicalDeviceDriverProperties driver_props;
        vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;
        vk::PhysicalDeviceShaderCorePropertiesAMD amd_shader_core_props;
        vk::PhysicalDeviceShaderCoreProperties2AMD amd_shader_core_properties2_props;
        vk::PhysicalDeviceVulkan11Properties vk11_props;
        vk::PhysicalDeviceVulkan12Properties vk12_props;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;
        vk::PhysicalDeviceExternalMemoryHostPropertiesEXT external_memory_host_props;

        props2.pNext = &props3;
        props3.pNext = &subgroup_props;
        subgroup_props.pNext = &driver_props;
        driver_props.pNext = &vk11_props;
        vk11_props.pNext = &vk12_props;

        VkBaseOutStructure * last_struct = (VkBaseOutStructure *)&vk12_props;

        if (maintenance4_support) {
            last_struct->pNext = (VkBaseOutStructure *)&props4;
            last_struct = (VkBaseOutStructure *)&props4;
        }
        if (sm_builtins) {
            last_struct->pNext = (VkBaseOutStructure *)&sm_props;
            last_struct = (VkBaseOutStructure *)&sm_props;
        }
        if (amd_shader_core_properties) {
            last_struct->pNext = (VkBaseOutStructure *)&amd_shader_core_props;
            last_struct = (VkBaseOutStructure *)&amd_shader_core_props;
        }
        if (amd_shader_core_properties2) {
            last_struct->pNext = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
            last_struct = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
        }
        if (device->subgroup_size_control) {
            last_struct->pNext = (VkBaseOutStructure *)&subgroup_size_control_props;
            last_struct = (VkBaseOutStructure *)&subgroup_size_control_props;
        }

#if defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        if (coopmat2_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat2_props;
            last_struct = (VkBaseOutStructure *)&coopmat2_props;
        }
#endif

        if (device->integer_dot_product) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_props;
            last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_props;
        }

        vk::PhysicalDevicePushDescriptorPropertiesKHR push_descriptor_props;
        if (push_descriptor_ext) {
            last_struct->pNext = (VkBaseOutStructure *)&push_descriptor_props;
            last_struct = (VkBaseOutStructure *)&push_descriptor_props;
        }

        if (device->external_memory_host) {
            last_struct->pNext = (VkBaseOutStructure *)&external_memory_host_props;
            last_struct = (VkBaseOutStructure *)&external_memory_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        // Debug only: report this maxComputeWorkGroupCount[0] instead of the driver's, e.g.
        // 4294967295 (what RADV reports) to reproduce limit-dependent grid math on any GPU.
        // A value above the real limit lets an op that trusts it dispatch an invalid grid.
        if (const char * wg_x_max = getenv("GGML_VK_DEBUG_MAX_WG_COUNT_X")) {
            device->properties.limits.maxComputeWorkGroupCount[0] = (uint32_t) strtoul(wg_x_max, nullptr, 10);
        }
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        if (device->driver_id == vk::DriverId::eMoltenvk) {
            // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
            // is available in the Vulkan SDK.
            device->external_memory_host = false;
        }

        device->wavefronts_per_simd = amd_shader_core_properties ? amd_shader_core_props.wavefrontsPerSimd : 0;

        device->push_descriptors = push_descriptor_ext &&
                                   push_descriptor_props.maxPushDescriptors >= MAX_PARAMETER_COUNT &&
                                   getenv("GGML_VK_DISABLE_PUSH_DESCRIPTORS") == nullptr;
        if (device->push_descriptors) {
            VK_LOG_DEBUG("ggml_vulkan: Push descriptors enabled (max=" << push_descriptor_props.maxPushDescriptors << ")");
        }

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.data()).find("(DG1)") == std::string::npos) &&
                                getenv("GGML_VK_DISABLE_ASYNC") == nullptr;

        if (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        const char* GGML_VK_FORCE_MAX_ALLOCATION_SIZE = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        if (GGML_VK_FORCE_MAX_ALLOCATION_SIZE != nullptr) {
            device->max_memory_allocation_size = std::stoull(GGML_VK_FORCE_MAX_ALLOCATION_SIZE);
        } else if (maintenance4_support) {
            device->max_memory_allocation_size = std::min(props3.maxMemoryAllocationSize, props4.maxBufferSize);
        } else {
            device->max_memory_allocation_size = props3.maxMemoryAllocationSize;
        }

        const char* GGML_VK_FORCE_MAX_BUFFER_SIZE = getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        if (GGML_VK_FORCE_MAX_BUFFER_SIZE != nullptr) {
            device->max_buffer_size = std::stoull(GGML_VK_FORCE_MAX_BUFFER_SIZE);
        } else if (maintenance4_support) {
            device->max_buffer_size = props4.maxBufferSize;
        } else {
            device->max_buffer_size = device->max_memory_allocation_size;
        }

        const char* GGML_VK_SUBALLOCATION_BLOCK_SIZE = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        if (GGML_VK_SUBALLOCATION_BLOCK_SIZE != nullptr) {
            device->suballocation_block_size = std::stoull(GGML_VK_SUBALLOCATION_BLOCK_SIZE);
        } else {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_block_size = 1024*1024*1024;
        }
        device->suballocation_block_size = std::min(device->suballocation_block_size, device->max_memory_allocation_size);

        device->subgroup_size = subgroup_props.subgroupSize;
        device->subgroup_size_log2 = uint32_t(log2f(float(device->subgroup_size)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        if (sm_builtins) {
            device->shader_core_count = sm_props.shaderSMCount;
        } else if (amd_shader_core_properties2) {
            device->shader_core_count = amd_shader_core_properties2_props.activeComputeUnitCount;
        } else if (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_count = ggml_vk_intel_shader_core_count(device->physical_device);
        } else {
            device->shader_core_count = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shaderRoundingModeRTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eBasic);

        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic);
#ifdef __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        if (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = false;
        }
#endif
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);
#ifdef __APPLE__
        if (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = false;
        }
#endif
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eVote);

        // Debug/diagnostic overrides: force-disable individual subgroup features to
        // emulate weaker GPUs (e.g. Tesla P100) on a more capable card, so that
        // device-specific fallbacks (Flash Attention -> CPU, graph-split explosion)
        // can be reproduced locally. All default OFF (no effect) unless the env is set.
        if (getenv("GGML_VK_DISABLE_SUBGROUP_SHUFFLE"))    device->subgroup_shuffle    = false;
        if (getenv("GGML_VK_DISABLE_SUBGROUP_VOTE"))       device->subgroup_vote       = false;
        if (getenv("GGML_VK_DISABLE_SUBGROUP_ARITHMETIC")) device->subgroup_arithmetic = false;
        if (getenv("GGML_VK_DISABLE_SUBGROUP_BALLOT"))     device->subgroup_ballot     = false;
        if (getenv("GGML_VK_DISABLE_SUBGROUP_CLUSTERED"))  device->subgroup_clustered  = false;
        if (getenv("GGML_VK_DISABLE_SUBGROUP_BASIC"))      device->subgroup_basic      = false;

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_per_submit = 100;
        const char* GGML_VK_MAX_NODES_PER_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        if (GGML_VK_MAX_NODES_PER_SUBMIT != nullptr) {
            uint32_t max_nodes_per_submit = std::stoul(GGML_VK_MAX_NODES_PER_SUBMIT);
            device->max_nodes_per_submit = std::max(max_nodes_per_submit, 1u);
        }

        const bool force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != nullptr;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        if (!ggml_vk_khr_cooperative_matrix_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = false;
        }

        device->integer_dot_product = device->integer_dot_product && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated;

        // ggmlR fork fix (properties-primary + architecture exclusion): the gate
        // above trusts integerDotProduct4x8BitPackedSignedAccelerated. On NVIDIA
        // pre-Turing GPUs (Maxwell / Pascal, detected by lack of
        // VK_KHR_cooperative_matrix) this properties flag is a false positive:
        // the driver reports it TRUE, but there is no hardware DP4A instruction
        // (DP4A became a real HW instruction only with Turing), so the
        // integer-dot (MMQ) matmul path is *emulated* and ~10-13x slower than
        // the plain f16 dequant path. Measured on Tesla P100 (GP100): isolated
        // MUL_MAT q6_K 322 -> 3487 GFLOP/s, q8_0 329 -> 4370 GFLOP/s when
        // integer-dot is disabled; Qwen3.5-9B prefill 58 -> 118 tok/s. The
        // properties flag stays the primary gate for every other device; this
        // only excludes the one architecture class where it is documented to
        // lie. Broader than strictly GP100 (P40/GTX1080 on GP10x do have HW
        // DP4A) but those are also pre-Turing with a competitive f16 path, so
        // the simple architecture exclusion is the safe choice. Override with
        // GGML_VK_FORCE_INTEGER_DOT_PRODUCT=1 if a pre-Turing card really wants it.
        if (device->integer_dot_product &&
            device->architecture == vk_device_architecture::NVIDIA_PRE_TURING &&
            getenv("GGML_VK_FORCE_INTEGER_DOT_PRODUCT") == nullptr) {
            device->integer_dot_product = false;
        }

        // Diagnostic dump of the raw *Accelerated properties, gated behind an
        // env var. Lets us later narrow the exclusion to a precise
        // properties-only check (find a flag that is honestly false on GP100)
        // without another round of on-device reconnaissance.
        if (getenv("GGMLR_DUMP_INTDOT_PROPS") != nullptr) {
            const auto & p = shader_integer_dot_product_props;
            std::cerr << "[intdot] dev='" << device->properties.deviceName.data()
                      << "' arch=" << (int)device->architecture
                      << " 8BitSignedAccel="   << (int)p.integerDotProduct8BitSignedAccelerated
                      << " 8BitUnsignedAccel="  << (int)p.integerDotProduct8BitUnsignedAccelerated
                      << " 4x8PackedSignedAccel="   << (int)p.integerDotProduct4x8BitPackedSignedAccelerated
                      << " 4x8PackedUnsignedAccel=" << (int)p.integerDotProduct4x8BitPackedUnsignedAccelerated
                      << " 4x8PackedMixedAccel="    << (int)p.integerDotProduct4x8BitPackedMixedSignednessAccelerated
                      << " 16BitSignedAccel="   << (int)p.integerDotProduct16BitSignedAccelerated
                      << " 32BitSignedAccel="   << (int)p.integerDotProduct32BitSignedAccelerated
                      << std::endl;
        }

        device->min_imported_host_pointer_alignment = external_memory_host_props.minImportedHostPointerAlignment;

        device->max_workgroup_size_log2 = uint32_t(log2f(float(device->properties.limits.maxComputeWorkGroupInvocations)));

        std::vector<vk::QueueFamilyProperties> queue_family_props = device->physical_device.getQueueFamilyProperties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        const bool allow_graphics_queue = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != nullptr);
        const vk::QueueFlagBits graphics_flag = allow_graphics_queue ? (vk::QueueFlagBits)0 : vk::QueueFlagBits::eGraphics;
        const uint32_t compute_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eCompute, graphics_flag, -1, 1);
        const uint32_t transfer_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute | graphics_flag, compute_queue_family_index, 1);

        const float priorities[] = { 1.0f, 1.0f };
        device->single_queue = compute_queue_family_index == transfer_queue_family_index && queue_family_props[compute_queue_family_index].queueCount == 1;

        std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<const char *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.pNext = nullptr;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.pNext = nullptr;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.pNext = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.pNext = nullptr;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.pNext = &vk12_features;

        last_struct = (VkBaseOutStructure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.pNext = nullptr;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        if (internally_sync_support) {
            last_struct->pNext = (VkBaseOutStructure *)&internally_synchronized_queues_features;
            last_struct = (VkBaseOutStructure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.pNext = nullptr;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        if (pipeline_robustness) {
            last_struct->pNext = (VkBaseOutStructure *)&pl_robustness_features;
            last_struct = (VkBaseOutStructure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        VkPhysicalDeviceMemoryPriorityFeaturesEXT memory_priority_features;
        memory_priority_features.pNext = nullptr;
        memory_priority_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
        memory_priority_features.memoryPriority = VK_FALSE;
        if (device->memory_priority) {
            last_struct->pNext = (VkBaseOutStructure *)&memory_priority_features;
            last_struct = (VkBaseOutStructure *)&memory_priority_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_size_control_features;
        subgroup_size_control_features.pNext = nullptr;
        subgroup_size_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        subgroup_size_control_features.computeFullSubgroups = false;
        subgroup_size_control_features.subgroupSizeControl = false;

        if (device->subgroup_size_control) {
            last_struct->pNext = (VkBaseOutStructure *)&subgroup_size_control_features;
            last_struct = (VkBaseOutStructure *)&subgroup_size_control_features;
        }

#if defined(VK_KHR_cooperative_matrix)
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
        coopmat_features.pNext = nullptr;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopmat_features.cooperativeMatrix = VK_FALSE;

        if (device->coopmat_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat_features;
            last_struct = (VkBaseOutStructure *)&coopmat_features;
        }
#endif

#if defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.pNext = nullptr;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
        if (coopmat2_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat2_features;
            last_struct = (VkBaseOutStructure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#endif

        VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV coopmat2_decode_vector_features {};
        coopmat2_decode_vector_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV;
        if (coopmat2_decode_vector_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
            last_struct = (VkBaseOutStructure *)&coopmat2_decode_vector_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME);
        }

#if defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.pNext = nullptr;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        if (bfloat16_support) {
            last_struct->pNext = (VkBaseOutStructure *)&bfloat16_features;
            last_struct = (VkBaseOutStructure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#endif

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        if (ocp_microscaling_extension) {
            last_struct->pNext = (VkBaseOutStructure *)&ocp_microscaling_features;
            last_struct = (VkBaseOutStructure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        if (shader_float8_extension) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_float8_features;
            last_struct = (VkBaseOutStructure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        if (maintenance4_support) {
            last_struct->pNext = (VkBaseOutStructure *)&maint4_features;
            last_struct = (VkBaseOutStructure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};
        shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
        if (device->integer_dot_product) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;
            last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        // ggmlR extension: float atomics, needed by the SSM backward kernels
        // that accumulate into gradients shared across workgroups.
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features {};
        atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        if (atomic_float_ext) {
            last_struct->pNext = (VkBaseOutStructure *)&atomic_float_features;
            last_struct = (VkBaseOutStructure *)&atomic_float_features;
            device_extensions.push_back("VK_EXT_shader_atomic_float");
        }

        VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE;
        if (dot2_f16_support) {
            last_struct->pNext = (VkBaseOutStructure *)&dot2_features;
            last_struct = (VkBaseOutStructure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        if (pipeline_executable_properties_support) {
            last_struct->pNext = (VkBaseOutStructure *)&pep_features;
            last_struct = (VkBaseOutStructure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        if (device->push_descriptors) {
            device_extensions.push_back("VK_KHR_push_descriptor");
        }

        if (device->external_memory_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

        // ggmlR Tensor Parallelism (P2P): enable opaque-fd external memory so
        // weight-split buffers can be exported on the owning device and imported
        // on the consumer device (device<->device transfer). Not in upstream ggml.
        if (device->external_memory_fd) {
            device_extensions.push_back("VK_KHR_external_memory");
            device_extensions.push_back("VK_KHR_external_memory_fd");
        }

#if defined(VK_EXT_shader_64bit_indexing)
        VkPhysicalDeviceShader64BitIndexingFeaturesEXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES_EXT;
        if (device->shader_64b_indexing) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_64bit_indexing_features;
            last_struct = (VkBaseOutStructure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#endif

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        if (device->device_fault) {
            last_struct->pNext = (VkBaseOutStructure *)&fault_features;
            last_struct = (VkBaseOutStructure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::DeviceQueueCreateFlags queue_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::DeviceQueueCreateFlags();

        if (compute_queue_family_index != transfer_queue_family_index) {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_index, 1, priorities});
            device_queue_create_infos.push_back({queue_flags, transfer_queue_family_index, 1, priorities + 1});
        } else if(!device->single_queue) {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_index, 2, priorities});
        } else {
            device_queue_create_infos.push_back({queue_flags, compute_queue_family_index, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#if defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#else
        device->bf16 = false;
#endif

        device->dot2_f16 = dot2_f16_support && dot2_features.shaderMixedFloatDotProductFloat16AccFloat32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !getenv("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_add = vk12_props.shaderRoundingModeRTEFloat16 &&
                            device->properties.limits.maxPushConstantsSize >= sizeof(vk_op_multi_add_push_constants) &&
                            getenv("GGML_VK_DISABLE_MULTI_ADD") == nullptr;

        device->supports_256_push_constants = device->properties.limits.maxPushConstantsSize >= 256;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->buffer_device_address = vk12_features.bufferDeviceAddress;
        device->vulkan_memory_model = vk12_features.vulkanMemoryModel;

        // ggmlR: the extension may be present while the specific atomic-add
        // capability is not, so both are required. An env override is kept for
        // bisecting a suspected atomics bug without rebuilding.
        device->atomic_float_add = atomic_float_ext &&
                                   atomic_float_features.shaderBufferFloat32AtomicAdd &&
                                   getenv("GGML_VK_DISABLE_ATOMIC_FLOAT") == nullptr;

        if (device->subgroup_size_control) {
            device->subgroup_min_size = subgroup_size_control_props.minSubgroupSize;
            device->subgroup_max_size = subgroup_size_control_props.maxSubgroupSize;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_size_control = device->subgroup_size_control &&
                (subgroup_size_control_props.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute) &&
                subgroup_size_control_features.subgroupSizeControl;

        device->subgroup_require_full_support = subgroup_size_control_features.computeFullSubgroups;

#if defined(VK_KHR_cooperative_matrix)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperativeMatrix;

        // coopmat1 fa shader currently assumes 32 invocations per subgroup
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_full_support &&
                                      device->subgroup_size_control && device->subgroup_min_size <= 32 &&
                                      device->subgroup_max_size >= 32;
#endif

        if (coopmat2_support) {
#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            if (coopmat2_features.cooperativeMatrixWorkgroupScope &&
                coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                coopmat2_features.cooperativeMatrixReductions &&
                coopmat2_features.cooperativeMatrixConversions &&
                coopmat2_features.cooperativeMatrixPerElementOperations &&
                coopmat2_features.cooperativeMatrixTensorAddressing &&
                coopmat2_features.cooperativeMatrixBlockLoads &&
                vk12_features.bufferDeviceAddress) {

                std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV
                    _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV =
                        (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, nullptr);

                VkCooperativeMatrixFlexibleDimensionsPropertiesNV empty_prop {};
                empty_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, empty_prop);

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, flexible_dimensions.data());

                bool found_fp16_128 = false,
                     found_fp16_256 = false,
                     found_fp32_128 = false,
                     found_fp32_256 = false;
                bool found_bf16_128 = false,
                     found_bf16_256 = false;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                for (auto &prop : flexible_dimensions) {
                    if (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        if (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            if (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                if (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    found_fp16_128 = true;
                                }
                                if (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    found_fp32_128 = true;
                                }
                            }
                            if (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                if (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    found_fp16_256 = true;
                                }
                                if (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    found_fp32_256 = true;
                                }
                            }
                        }

#if defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        if (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            if (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                found_bf16_128 = true;
                            }
                            if (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                found_bf16_256 = true;
                            }
                        }
#endif
                    }
                }
                if (found_fp16_128 && found_fp16_256 &&
                    found_fp32_128 && found_fp32_256 &&
                    coopmat2_props.cooperativeMatrixFlexibleDimensionsMaxDimension >= 512) {
                    device->coopmat2 = true;
                    device->coopmat2_bf16_support = found_bf16_128 && found_bf16_256;
                    device->coopmat2_decode_vector = coopmat2_decode_vector_support && coopmat2_decode_vector_features.cooperativeMatrixDecodeVector;
                }
            }
#endif
        }

        if (!vk11_features.storageBuffer16BitAccess) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::endl;
            throw std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#ifdef GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#endif

        if (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#if defined(VK_KHR_cooperative_matrix)
        if (device->coopmat_support) {
            // Query supported shapes
            std::vector<VkCooperativeMatrixPropertiesKHR> cm_props;

            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
                (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_num;

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_num, nullptr);

            cm_props.resize(cm_props_num);

            for (auto& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_num, cm_props.data());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            for (auto& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.ResultType) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                if ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    if ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        if (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = true;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = true;
                        }
                        if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = true;
                        }
                    } else if ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        if (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = true;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = true;
                        }
                        if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = true;
                        }
                    }
                } else if ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = true;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#if defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                if (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    if (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = true;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = true;
                    }
                }
#endif
            }

            if (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = false;
            }
        }

        if (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#endif
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .setFlags(vk::DeviceCreateFlags())
            .setQueueCreateInfos(device_queue_create_infos)
            .setPEnabledExtensionNames(device_extensions);
        device_create_info.setPNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        if (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_queue = ggml_vk_create_queue(device, compute_queue_family_index, 0, { vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer }, false);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        for (uint32_t i = 0; i < GGML_TYPE_COUNT; ++i) {
            switch (device->vendor_id) {
            case VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = true;
                device->mul_mat_s[i]    = true;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                break;
            case VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                const bool use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = true;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                break;
            }
            case VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = false;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = false;
                break;
            case VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                break;
            case VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = true;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                break;
            default:
                device->mul_mat_l[i] = true;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = true;
                device->mul_mat_id_l[i] = true;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                break;
            }

#if VK_HEADER_VERSION >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            if (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = false;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = false;
            }
#endif

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        for (uint32_t i = 0; i < MAX_PARAMETER_COUNT; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::DescriptorSetLayoutBindingFlagsCreateInfo dslbfci = { dsl_binding_flags };

        vk::DescriptorSetLayoutCreateFlags dsl_flags = {};
        if (device->push_descriptors) {
            dsl_flags |= vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
        }
        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
            dsl_flags,
            dsl_binding);
        descriptor_set_layout_create_info.setPNext(&dslbfci);
        device->dsl = device->device.createDescriptorSetLayout(descriptor_set_layout_create_info);

        ggml_vk_load_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        const bool prefers_transfer_queue =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_queue;

        if (!device->single_queue) {
            const uint32_t transfer_queue_index = compute_queue_family_index == transfer_queue_family_index ? 1 : 0;
            device->transfer_queue = ggml_vk_create_queue(device, transfer_queue_family_index, transfer_queue_index, { vk::PipelineStageFlagBits::eTransfer }, true);

            device->async_use_transfer_queue = prefers_transfer_queue || (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != nullptr);
        } else {
            device->transfer_queue = ggml_vk_create_aliased_queue(device, device->compute_queue);

            device->async_use_transfer_queue = false;
        }

        device->fence = device->device.createFence({});

        device->buffer_type = {
            /* .iface    = */ ggml_backend_vk_buffer_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ new ggml_backend_vk_buffer_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = getenv("GGML_VK_SERIALIZE_SUBMISSIONS") != nullptr;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != nullptr;

        device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != nullptr;

        device->add_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.minStorageBufferOffsetAlignment);

        device->mmvq_mode = 0;
        if (getenv("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_mode = -1;
        } else if (getenv("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_mode = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        return device;
    }

    return vk_instance.devices[idx];
}

