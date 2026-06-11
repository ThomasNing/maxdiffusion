import torch
from diffusers import FluxPipeline
import numpy as np
import os

# 1. Load the Pipeline (CPU strictly)
# FLUX.2-klein-4B is assumed to be compatible with typical diffusers FluxPipeline.
# We use float32 for CPU execution for highest precision to compare with Jax.
print("Loading FLUX.2-klein-4B pipeline on CPU...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.float32,
)
pipe.to("cpu")

# 2. Setup Hooks to capture intermediate layers
intermediates = {}
def get_hook(name):
    def hook(module, module_in, module_out):
        # module_out might be a tuple (hidden_states, encoder_hidden_states), so we unpack or save as is
        if isinstance(module_out, tuple):
            intermediates[f"{name}_hidden_states"] = module_out[0].detach().cpu().numpy()
            if len(module_out) > 1 and module_out[1] is not None:
                intermediates[f"{name}_encoder_hidden_states"] = module_out[1].detach().cpu().numpy()
        elif isinstance(module_out, torch.Tensor):
            intermediates[name] = module_out.detach().cpu().numpy()
    return hook

# Hook double-stream transformer blocks
if hasattr(pipe.transformer, 'transformer_blocks'):
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        block.register_forward_hook(get_hook(f"double_block_{i}"))

# Hook single-stream transformer blocks
if hasattr(pipe.transformer, 'single_transformer_blocks'):
    for i, block in enumerate(pipe.transformer.single_transformer_blocks):
        block.register_forward_hook(get_hook(f"single_block_{i}"))

print("Hooks registered for transformer blocks.")

# 3. Capture Initial Latent Noise via callback
initial_latents = None

def callback_on_step_end(pipe, step, timestep, callback_kwargs):
    global initial_latents
    if step == 0:
        # Save the very first latents before they are updated
        initial_latents = callback_kwargs["latents"].detach().cpu().numpy()
        print("Captured initial latents at step 0.")
    return callback_kwargs

# 4. Run single inference step
prompt = "A golden retriever playing with a ball in the park"
seed = 42
generator = torch.Generator(device="cpu").manual_seed(seed)

print(f"Running inference for prompt: '{prompt}'")
# We use num_inference_steps=1 or 2 so it doesn't take forever but still hits the transformer
with torch.no_grad():
    output = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=1,
        generator=generator,
        guidance_scale=0.0, # minimal processing for pure model checks
        callback_on_step_end=callback_on_step_end
    )
print("Inference completed.")

# 5. Save everything to disk
output_dir = "flux2_klein_4b_intermediates"
os.makedirs(output_dir, exist_ok=True)

# Save initial latents 
np.save(os.path.join(output_dir, "initial_latents.npy"), initial_latents)

# Save all intermediates from the hooks
for name, array in intermediates.items():
    np.save(os.path.join(output_dir, f"{name}.npy"), array)

print(f"All {len(intermediates)} intermediate layers and initial latents saved to {output_dir}/")
