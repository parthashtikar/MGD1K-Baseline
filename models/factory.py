from .dram_universal import DRAMNetUniversal


def make_dram(
    mode="both",
    fusion="none",
    width=32,
    use_channel_attn=True,
    attn_kernel=7,
    in_ch=1,
    out_ch=2,
):
    """
    Convenience wrapper to instantiate DRAMNetUniversal with our common defaults.
    """
    return DRAMNetUniversal(
        in_ch=in_ch,
        out_ch=out_ch,
        width=width,
        mode=mode,
        fusion=fusion,
        use_channel_attn=use_channel_attn,
        attn_kernel=attn_kernel,
    )
