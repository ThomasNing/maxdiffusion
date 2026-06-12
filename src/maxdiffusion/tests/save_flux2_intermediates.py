import os
import torch
import numpy as np
from diffusers import Flux2KleinPipeline

# Unified dictionary to hold every diagnostic array
saved_data = {}

# ==========================================
# 1. Load Model (Strictly CPU & Float32)
# ==========================================
print("Loading FLUX.2-klein-4B pipeline on CPU...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.float32,
)
pipe.to("cpu")

transformer = pipe.transformer
vae = pipe.vae

# ==========================================
# 2. Register Hooks for FLUX.2 Core Elements
# ==========================================
print("Registering updated hooks for FLUX.2 conditioning layouts...")

# --- Target 1) Text Embeddings Sequence ---
def hook_context_embedder(module, args, output):
    saved_data["sequence_text_emb"] = output.detach().cpu().numpy()

transformer.context_embedder.register_forward_hook(hook_context_embedder)

# --- Target 2) & 3) Time/Guidance Conditioning Vector ---
# Flux.2 replaces time_text_embed with time_guidance_embed
def hook_time_guidance_embed(module, args, output):
    saved_data["joint_time_guidance_conditioning_vector_t0"] = output.detach().cpu().numpy()

if hasattr(transformer, "time_guidance_embed"):
    transformer.time_guidance_embed.register_forward_hook(hook_time_guidance_embed)
    
    # Capture pure sinusoidal time embedding before guidance mix
    if hasattr(transformer.time_guidance_embed, "timestep_embedder"):
        transformer.time_guidance_embed.timestep_embedder.register_forward_hook(
            lambda m, inp, out: saved_data.update({"pure_time_embedding": out.detach().cpu().numpy()})
        )

# --- Target 4) Global Shift/Scale Parameter Generators ---
# These are the global blocks generating modulation keys for the whole network
if hasattr(transformer, "double_stream_modulation_img"):
    transformer.double_stream_modulation_img.register_forward_hook(
        lambda m, inp, out: saved_data.update({"global_double_img_modulation_params": out.detach().cpu().numpy()})
    )
if hasattr(transformer, "double_stream_modulation_txt"):
    transformer.double_stream_modulation_txt.register_forward_hook(
        lambda m, inp, out: saved_data.update({"global_double_txt_modulation_params": out.detach().cpu().numpy()})
    )
if hasattr(transformer, "single_stream_modulation"):
    transformer.single_stream_modulation.register_forward_hook(
        lambda m, inp, out: saved_data.update({"global_single_joint_modulation_params": out.detach().cpu().numpy()})
    )

# --- Target 5) Top-level Latent Entry ---
def transformer_top_pre_hook(module, args, kwargs):
    if "initial_transformer_input_latents" not in saved_data:
        h = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        if h is not None:
            saved_data["initial_transformer_input_latents"] = h.detach().cpu().numpy()

transformer.register_forward_pre_hook(transformer_top_pre_hook, with_kwargs=True)


# ==========================================
# 3. Register Block-by-Block Hooks (4, 5, 6)
# ==========================================
print("Registering deep hooks inside individual transformer layers...")

# --- Hook Double Stream Transformer Blocks ---
if hasattr(transformer, "transformer_blocks"):
    for i, block in enumerate(transformer.transformer_blocks):
        
        # Capture input latents per block boundary
        def make_double_pre_hook(block_idx):
            def pre_hook(module, args, kwargs):
                h = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                ctx = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
                if h is not None:
                    saved_data[f"double_block_{block_idx}_input_image_latents"] = h.detach().cpu().numpy()
                if ctx is not None:
                    saved_data[f"double_block_{block_idx}_input_text_latents"] = ctx.detach().cpu().numpy()
            return pre_hook
        block.register_forward_pre_hook(make_double_pre_hook(i), with_kwargs=True)

        # Target 6) Capture features right after scaling/shifting is applied inside norms
        if hasattr(block, "norm1"):
            block.norm1.register_forward_hook(
                lambda m, inp, out, idx=i: saved_data.update({f"double_block_{idx}_modulated_image_latents": out.detach().cpu().numpy()})
            )
        if hasattr(block, "norm1_context"):
            block.norm1_context.register_forward_hook(
                lambda m, inp, out, idx=i: saved_data.update({f"double_block_{idx}_modulated_text_latents": out.detach().cpu().numpy()})
            )

        # Capture final outputs exiting the block
        def make_double_post_hook(block_idx):
            def post_hook(module, args, output):
                if isinstance(output, tuple):
                    saved_data[f"double_block_{block_idx}_output_image_latents"] = output[0].detach().cpu().numpy()
                    if len(output) > 1 and output[1] is not None:
                        saved_data[f"double_block_{block_idx}_output_text_latents"] = output[1].detach().cpu().numpy()
            return post_hook
        block.register_forward_hook(make_double_post_hook(i))

# --- Hook Single Stream Transformer Blocks ---
if hasattr(transformer, "single_transformer_blocks"):
    for i, block in enumerate(transformer.single_transformer_blocks):
        
        block.register_forward_pre_hook(
            lambda m, args, kwargs, idx=i: saved_data.update({f"single_block_{idx}_input_latents": (kwargs.get("hidden_states", args[0] if len(args) > 0 else None)).detach().cpu().numpy()}),
            with_kwargs=True
        )
        # Target 6) Modulated joint inputs
        if hasattr(block, "norm"):
            block.norm.register_forward_hook(
                lambda m, inp, out, idx=i: saved_data.update({f"single_block_{idx}_modulated_latents": out.detach().cpu().numpy()})
            )
        # Capture outputs exiting the single block
        block.register_forward_hook(
            lambda m, inp, out, idx=i: saved_data.update({f"single_block_{idx}_output_latents": out.detach().cpu().numpy()})
        )


# ==========================================
# 4. Register VAE Decoder Hooks
# ==========================================
print("Registering hooks across VAE Decoder layers...")

vae.register_forward_pre_hook(
    lambda m, args, kwargs: saved_data.update({"vae_input_unpacked_scaled_latents": args[0].detach().cpu().numpy()}),
    with_kwargs=True
)

if hasattr(vae, "decoder"):
    decoder = vae.decoder
    if hasattr(decoder, "conv_in"):
        decoder.conv_in.register_forward_hook(lambda m, inp, out: saved_data.update({"vae_decoder_conv_in_output": out.detach().cpu().numpy()}))
    if hasattr(decoder, "mid_block") and decoder.mid_block is not None:
        decoder.mid_block.register_forward_hook(lambda m, inp, out: saved_data.update({"vae_decoder_mid_block_output": out.detach().cpu().numpy()}))
    if hasattr(decoder, "up_blocks"):
        for i, up_block in enumerate(decoder.up_blocks):
            up_block.register_forward_hook(lambda m, inp, out, idx=i: saved_data.update({f"vae_decoder_up_block_{idx}_output": out.detach().cpu().numpy()}))
    if hasattr(decoder, "conv_out"):
        decoder.conv_out.register_forward_hook(lambda m, inp, out: saved_data.update({"vae_decoder_conv_out_output": out.detach().cpu().numpy()}))


# ==========================================
# 5. Execute Pipeline Pass
# ==========================================
def callback_on_step_end(pipe, step, timestep, callback_kwargs):
    if step == 0:
        saved_data["initial_pipeline_latents"] = callback_kwargs["latents"].detach().cpu().numpy()
    return callback_kwargs

prompt = "A detailed vector illustration of a robotic hummingbird"
generator = torch.Generator(device="cpu").manual_seed(42)

print("Executing single-step forward pass...")
with torch.no_grad():
    _ = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=1,
        generator=generator,
        guidance_scale=0.0,
        callback_on_step_end=callback_on_step_end
    )

# ==========================================
# 6. Save Bundle to Disk
# ==========================================
output_filename = "flux2_klein_complete_diagnostic_bundle.npz"
np.savez_compressed(output_filename, **saved_data)

print(f"\n=======================================================")
print(f"SUCCESS! Harvested {len(saved_data)} clean tracking arrays.")
print(f"File Saved: {os.path.abspath(output_filename)}")
print(f"=======================================================")
