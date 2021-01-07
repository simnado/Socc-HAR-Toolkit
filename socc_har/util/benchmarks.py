phase_1 = dict(
    R2plus1D_34_head='narendorf/soccar-32-ph0/b10b8daa045f41958f70b2823b8fe9b3',
    SlowFast8x8_head='narendorf/soccar-32-ph0/9d53ec5e5225466aa4e9652a7be98d4f',
    SlowFast4x16_head='narendorf/soccar-32-ph0/acbf6bb99e69494792cf88b7d167a079',
    ir_CSN_head='narendorf/soccar-32-ph0/9a7c4b4803de43a7b15b3678cd32e3c0',

    R2plus1D_34_backbone='narendorf/soccar-32-ph1/a0f0001969894d0eb2375e6b347fb701',
    SlowFast_8x8_backbone='narendorf/soccar-32-ph1/b703c8fee3af4817ab347839901e0165',
    SlowFast_4x16_backbone='narendorf/soccar-32-ph1/e00640bb078a4dc4bdb451fd2da49afb',
    ir_CSN_4g_backbone='narendorf/soccar-32-ph1/822944fc50984cc99ee96586ec15e9f1',

    ir_CSN_verified='narendorf/soccar-32-ph1/578effe0da8e4aaa89f60eeaa3a0b55e'
)

phase_2 = dict(
    ir_CSN_verified_32_12='narendorf/socc-har-32-ph1/e2e01b66e31246419280edf18c20cf33',
    ir_CSN_verified_48_10='narendorf/socc-har-32-ph1/d82e3d6c98e545669627d3e0c0451899',

    # R2plus1D_34_32_2='narendorf/soccar-32-ph1/4488f5f18d7e438aad586eaa348edd60',
    # R2plus1D_34_64_1='narendorf/soccar-32-ph1/5517708f1c684506ae5afca3bb50e63e',
    # R2plus1D_34_48_2='narendorf/soccar-32-ph1/245fa59c071947d2a5b071dbc0eef395',

    # SlowFast_4x16_32_3='narendorf/soccar-32-ph1/a55cf7d521f044f998e9f24cd10b3a29',
    # SlowFast_4x16_48_2='narendorf/soccar-32-ph1/6fa6208f55724ff6b68e4b8e4e61b16b',

    # ir_CSN_4g_32_3='narendorf/soccar-32-ph1/b72644fe5e324948a6c8c5bdadd06ea6',
    # ir_CSN_4g_48_2='narendorf/soccar-32-ph1/ee68cb3f6dd14c66b2c68d1df72a818b',
)

phase_3 = dict(
    ir_CSN_verified_48_10_socc_har_28='narendorf/socc-har-32-ph2/7babc9c3f47d460b91f91e8cc8c768d7'
)

benchmarks = [phase_1, phase_2, phase_3]
