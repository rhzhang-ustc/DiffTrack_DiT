def build_decoder(config: dict, drop_path_allocator,
                  dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                  z_shape, x_shape):
    decoder_config = config['transformer']['decoder']
    decoder_type = decoder_config['type']
    
    if decoder_type == 'concatenation_feature_fusion':
        fusion = decoder_config['fusion']
        from .concatenated_fusion import build_feature_map_generation_decoder
        return build_feature_map_generation_decoder(config, drop_path_allocator,
                                                    dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                                    z_shape, x_shape, fusion)
    elif decoder_type == 'target_query_decoder':
        from .target_query_decoder import build_target_query_decoder
        return build_target_query_decoder(config, drop_path_allocator,
                                          dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                          z_shape, x_shape)
    elif decoder_type == 'diffusion_vit':
        from .dit_decoder import build_dit_decoder
        return build_dit_decoder(config, 
                                dim, num_heads, mlp_ratio,
                                z_shape, x_shape)

    else:
        raise NotImplementedError(decoder_type)
