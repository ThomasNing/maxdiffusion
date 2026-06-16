import unittest
import numpy as np
import jax
import jax.numpy as jnp
#from .. import pyconfig
from maxdiffusion.generate_flux2klein import load_or_generate_latents
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler

# -----------------------------------------------------------------------------
# Module-level Helper Functions for Packing & Coordinate IDs
# -----------------------------------------------------------------------------

def prepare_latent_image_ids(batch_size, height, width):
    """Generates 4D position coordinates (T, H, W, L) for latent tensors."""
    grid = jnp.zeros((height, width, 4), dtype=jnp.int32)
    grid = grid.at[..., 1].set(jnp.arange(height)[:, None])
    grid = grid.at[..., 2].set(jnp.arange(width)[None, :])
    latent_ids = grid.reshape(-1, 4)
    latent_ids = jnp.expand_dims(latent_ids, axis=0)
    latent_ids = jnp.repeat(latent_ids, batch_size, axis=0)
    return latent_ids

def pack_latents(latents):
    """[B, C, H, W] -> [B, H*W, C]"""
    batch_size, num_channels, height, width = latents.shape
    x = jnp.reshape(latents, (batch_size, num_channels, height * width))
    x = jnp.transpose(x, (0, 2, 1))
    return x

def unpack_latents_with_ids(x, x_ids, height, width):
    """[B, H*W, C] -> [B, C, H, W] using coordinate IDs."""
    batch_size, seq_len, ch = x.shape
    x_list = []
    for b in range(batch_size):
        data = x[b]
        pos = x_ids[b]
        h_ids = pos[:, 1].astype(jnp.int32)
        w_ids = pos[:, 2].astype(jnp.int32)
        flat_ids = h_ids * width + w_ids
        out = jnp.zeros((height * width, ch), dtype=x.dtype)
        out = out.at[flat_ids].set(data)
        out = jnp.transpose(jnp.reshape(out, (height, width, ch)), (2, 0, 1))
        x_list.append(out)
    return jnp.stack(x_list, axis=0)

def unpatchify_latents(latents):
    """Reverses the 2x2 spatial patch grouping: [B, C, H, W] -> [B, C/4, H*2, W*2]"""
    batch_size, num_channels_latents, height, width = latents.shape
    x = jnp.reshape(latents, (batch_size, num_channels_latents // 4, 2, 2, height, width))
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    x = jnp.reshape(x, (batch_size, num_channels_latents // 4, height * 2, width * 2))
    return x

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)


class GenerateFlux2KleinTest(unittest.TestCase):

    def test_generate_random_latents_shape(self):
        config = {
            "use_latents": False,
            "batch_size": 2,
            "height": 1024,
            "width": 512,
        }
        latents = load_or_generate_latents(config)
        
        expected_shape = (2, 32, 1024 // 8, 512 // 8)
        self.assertEqual(latents.shape, expected_shape)

    def test_load_golden_latents_shape(self):
        # This test assumes `flux2_klein_complete_diagnostic_bundle.npz` is in the execution directory
        # We will test using batch=1, height=512, width=512 which matches the diagnostic generator
        config = {
            "use_latents": True,
            "batch_size": 1,
            "height": 512,
            "width": 512,
        }
        try:
            latents = load_or_generate_latents(config)
            expected_shape = (1, 32, 512 // 8, 512 // 8)
            self.assertEqual(latents.shape, expected_shape)
        except FileNotFoundError:
            self.skipTest("Skipping test_load_golden_latents_shape because flux2_klein_complete_diagnostic_bundle.npz is not present.")

    def test_qwen3_prompt_embeddings(self):
        from maxdiffusion.generate_flux2klein import encode_prompt
        prompt = "A detailed vector illustration of a robotic hummingbird"
        
        print("Running test_qwen3_prompt_embeddings...")
        try:
            embeds = encode_prompt(prompt)
            expected_shape = (1, 512, 7680)
            
            self.assertIsInstance(embeds, np.ndarray)
            self.assertEqual(embeds.shape, expected_shape)
            self.assertNotEqual(np.sum(np.abs(embeds)), 0.0, "Embeddings should not be all zeros.")
            print("Successfully verified prompt embeddings shape and non-zero contents!")
        except Exception as e:
            self.fail(f"Failed to generate prompt embeddings: {e}")

    def test_context_embedder_projection(self):
        import os
        import torch
        import jax
        import jax.numpy as jnp
        import flax
        from flax.linen import partitioning as nn_partitioning
        from jax.sharding import Mesh
        from safetensors.torch import load_file
        
        from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
        from maxdiffusion.generate_flux2klein import encode_prompt
        from maxdiffusion import pyconfig
        from maxdiffusion.max_utils import create_device_mesh
        
        # 1. Initialize pyconfig if needed
        if getattr(pyconfig, "config", None) is None:
            pyconfig.initialize([
                None,
                "src/maxdiffusion/configs/base_flux_dev.yml",
                "run_name=flux_test",
                "output_dir=/tmp/",
                "jax_cache_dir=/tmp/cache_dir",
            ], unittest=True)
        config = pyconfig.config
        
        # 2. Setup device mesh
        try:
            devices_array = create_device_mesh(config)
            mesh = Mesh(devices_array, config.mesh_axes)
        except Exception as e:
            self.skipTest(f"Skipping because device mesh creation failed (might not be running on TPU VM): {e}")
            
        # 3. Locate safetensors
        cache_dir = "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"
        if not os.path.exists(cache_dir):
            self.skipTest("Skipping because Hugging Face cache directory is not present.")
            
        snapshots = os.listdir(cache_dir)
        if not snapshots:
            self.skipTest("Skipping because no snapshot found in cache.")
        snapshot_dir = os.path.join(cache_dir, snapshots[0])
        safetensors_path = os.path.join(snapshot_dir, "transformer", "diffusion_pytorch_model.safetensors")
        
        if not os.path.exists(safetensors_path):
            self.skipTest(f"Skipping because safetensors file not found: {safetensors_path}")
            
        # 4. Load PyTorch weight and convert
        pt_state_dict = load_file(safetensors_path)
        if "context_embedder.weight" not in pt_state_dict:
            self.fail("context_embedder.weight not found in transformer safetensors!")
        pt_weight = pt_state_dict["context_embedder.weight"]
        jax_weight = jnp.array(pt_weight.to(torch.float32).cpu().numpy().T)
        
        # 5. Instantiate model with Klein config
        transformer = FluxTransformer2DModel(
            in_channels=128,
            num_layers=5,
            num_single_layers=20,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=7680,
            mlp_ratio=3.0,
            qkv_bias=False,
            joint_attention_bias=False,
            x_embedder_bias=False,
            proj_out_bias=False,
            mesh=mesh,
        )
        
        # 6. Initialize and run forward pass within mesh context
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            batch_size = 1
            seq_len_img = 256
            seq_len_txt = 512
            
            img = jnp.zeros((batch_size, seq_len_img, 128))
            img_ids = jnp.zeros((batch_size, seq_len_img, 3))
            txt = jnp.zeros((batch_size, seq_len_txt, 7680))
            txt_ids = jnp.zeros((batch_size, seq_len_txt, 3))
            vec = jnp.zeros((batch_size, 768))
            t_vec = jnp.zeros((batch_size,))
            guidance_vec = jnp.zeros((batch_size,))
            
            key = jax.random.PRNGKey(0)
            variables = transformer.init(
                key,
                hidden_states=img,
                img_ids=img_ids,
                encoder_hidden_states=txt,
                txt_ids=txt_ids,
                pooled_projections=vec,
                timestep=t_vec,
                guidance=guidance_vec,
            )
            params = variables["params"]
            
            # Unbox LogicallyPartitioned parameters to get raw JAX arrays
            import flax.linen.spmd as flax_spmd
            params = jax.tree_util.tree_map(
                lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
                params,
                is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
            )
            
            # Replace JAX weight
            params = flax.core.unfreeze(params)
            params["txt_in"]["kernel"] = jax_weight
            params = flax.core.freeze(params)
            
            # Encode prompt
            prompt = "A detailed vector illustration of a robotic hummingbird"
            prompt_embeds = encode_prompt(prompt)
            prompt_embeds_jax = jnp.array(prompt_embeds)
            
            # Run projection
            projected = transformer.apply(
                {"params": params},
                prompt_embeds_jax,
                method=lambda self, x: self.txt_in(x)
            )
            
        # 7. Load golden projected embeddings
        bundle_path = "src/maxdiffusion/tests/flux2_klein_complete_diagnostic_bundle.npz"
        if not os.path.exists(bundle_path):
            self.skipTest(f"Skipping because diagnostic bundle not found: {bundle_path}")
            
        bundle = np.load(bundle_path)
        if "sequence_text_emb" not in bundle:
            self.fail("sequence_text_emb key not found in diagnostic bundle!")
        golden_projected = bundle["sequence_text_emb"]
        golden_projected_jax = jnp.array(golden_projected)
        
        # 8. Assert close within tolerance (rtol=1e-2, atol=0.8)
        np.testing.assert_allclose(projected, golden_projected_jax, rtol=1e-2, atol=8e-1)

    def test_packing_roundtrip_parity(self):
        """Verify JAX latent patchify -> pack -> unpack -> unpatchify matches exactly."""
        # Start with random unpacked latents: shape (1, 32, 64, 64)
        key = jax.random.PRNGKey(0)
        initial_latents = jax.random.normal(key, (1, 32, 64, 64))
        
        def patchify_latents(latents):
            batch_size, num_channels, height, width = latents.shape
            x = jnp.reshape(latents, (batch_size, num_channels, height // 2, 2, width // 2, 2))
            x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
            x = jnp.reshape(x, (batch_size, num_channels * 4, height // 2, width // 2))
            return x
            
        patchified = patchify_latents(initial_latents)
        self.assertEqual(patchified.shape, (1, 128, 32, 32))
        
        packed = pack_latents(patchified)
        self.assertEqual(packed.shape, (1, 1024, 128))
        
        latent_ids = prepare_latent_image_ids(batch_size=1, height=32, width=32)
        
        unpacked = unpack_latents_with_ids(packed, latent_ids, height=32, width=32)
        self.assertEqual(unpacked.shape, (1, 128, 32, 32))
        
        np.testing.assert_array_equal(np.array(unpacked), np.array(patchified))
        
        unpatchified = unpatchify_latents(unpacked)
        self.assertEqual(unpatchified.shape, (1, 32, 64, 64))
        
        np.testing.assert_allclose(
            np.array(unpatchified),
            np.array(initial_latents),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Full latent round-trip failed!"
        )

    def test_scheduler_timesteps_parity(self):
        """Verify JAX FlaxFlowMatchScheduler timesteps/sigmas match PyTorch exactly."""
        try:
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler
        except ImportError:
            self.skipTest("PyTorch/diffusers not available. Run on TPU VM.")
            
        pytorch_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            time_shift_type="exponential",
        )
        
        for steps in [4, 10, 28, 50]:
            image_seq_len = 1024
            mu = compute_empirical_mu(image_seq_len, steps)
            pytorch_scheduler.set_timesteps(num_inference_steps=steps, mu=mu, device="cpu")
            
            py_timesteps = pytorch_scheduler.timesteps.numpy()
            py_sigmas = pytorch_scheduler.sigmas.numpy()
            
            jax_scheduler = FlaxFlowMatchScheduler(
                num_train_timesteps=1000,
                shift=mu,
                sigma_max=1.0,
                sigma_min=0.001,
                inverse_timesteps=False,
                extra_one_step=False,
                reverse_sigmas=False,
                use_dynamic_shifting=True,
                time_shift_type="exponential",
            )
            
            state = jax_scheduler.create_state()
            state = jax_scheduler.set_timesteps_ltx2(
                state=state,
                num_inference_steps=steps,
                shift=mu,
            )
            
            jax_timesteps = np.array(state.timesteps)
            jax_sigmas = np.array(state.sigmas)
            
            np.testing.assert_allclose(
                jax_timesteps,
                py_timesteps,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Timestep mismatch for steps={steps}!"
            )
            
            np.testing.assert_allclose(
                jax_sigmas,
                py_sigmas[:-1],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Sigma mismatch for steps={steps}!"
            )

    def test_attention_blocks_parity(self):
        """Verifies that JAX joint-attention (double) and single-stream blocks match PyTorch golden outputs."""
        import os
        import torch
        import jax
        import jax.numpy as jnp
        import flax
        from flax.linen import partitioning as nn_partitioning
        from jax.sharding import Mesh
        from safetensors.torch import load_file
        
        from maxdiffusion.models.flux.transformers.transformer_flux_flax import FluxTransformer2DModel
        from maxdiffusion import pyconfig
        from maxdiffusion.max_utils import create_device_mesh
        
        # 1. Initialize pyconfig if needed
        if getattr(pyconfig, "config", None) is None:
            pyconfig.initialize([
                None,
                "src/maxdiffusion/configs/base_flux_dev.yml",
                "run_name=flux_test",
                "output_dir=/tmp/",
                "jax_cache_dir=/tmp/cache_dir",
            ], unittest=True)
        config = pyconfig.config
        
        # 2. Setup device mesh
        try:
            devices_array = create_device_mesh(config)
            mesh = Mesh(devices_array, config.mesh_axes)
        except Exception as e:
            self.skipTest(f"Skipping because device mesh creation failed: {e}")
            
        # 3. Locate safetensors
        cache_dir = "/mnt/data/hf_cache/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"
        if not os.path.exists(cache_dir):
            self.skipTest("Skipping because Hugging Face cache directory is not present.")
            
        snapshots = os.listdir(cache_dir)
        if not snapshots:
            self.skipTest("Skipping because no snapshot found in cache.")
        snapshot_dir = os.path.join(cache_dir, snapshots[0])
        safetensors_path = os.path.join(snapshot_dir, "transformer", "diffusion_pytorch_model.safetensors")
        
        if not os.path.exists(safetensors_path):
            self.skipTest(f"Skipping because safetensors file not found: {safetensors_path}")
            
        # 4. Load PyTorch weight and golden diagnostic bundle
        print("Loading weights and golden intermediates...")
        pt_state_dict = load_file(safetensors_path)
        
        bundle_path = "src/maxdiffusion/tests/flux2_klein_complete_diagnostic_bundle.npz"
        if not os.path.exists(bundle_path):
            self.skipTest(f"Skipping because diagnostic bundle not found: {bundle_path}")
        bundle = np.load(bundle_path)
        
        # 5. Instantiate model with Klein config, global modulation, and SwiGLU enabled!
        print("Instantiating JAX FluxTransformer2DModel...")
        transformer = FluxTransformer2DModel(
            in_channels=128,
            num_layers=5,
            num_single_layers=20,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=7680,
            mlp_ratio=3.0,
            qkv_bias=False,
            joint_attention_bias=False,
            x_embedder_bias=False,
            proj_out_bias=False,
            use_global_modulation=True, # Enable global modulation!
            use_swiglu=True,             # Enable SwiGLU!
            axes_dims_rope=(32, 32, 32, 32), # Configure 4D RoPE!
            theta=2000,                  # Align positional embeddings base theta!
            mesh=mesh,
        )
        
        # 6. Initialize JAX parameters within mesh context
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            batch_size = 1
            seq_len_img = 256
            seq_len_txt = 512
            
            img = jnp.zeros((batch_size, seq_len_img, 128))
            img_ids = jnp.zeros((batch_size, seq_len_img, 4)) # 4D coords!
            txt = jnp.zeros((batch_size, seq_len_txt, 7680))
            txt_ids = jnp.zeros((batch_size, seq_len_txt, 4)) # 4D coords!
            vec = jnp.zeros((batch_size, 768))
            t_vec = jnp.zeros((batch_size,))
            guidance_vec = jnp.zeros((batch_size,))
            
            key = jax.random.PRNGKey(0)
            variables = transformer.init(
                key,
                hidden_states=img,
                img_ids=img_ids,
                encoder_hidden_states=txt,
                txt_ids=txt_ids,
                pooled_projections=vec,
                timestep=t_vec,
                guidance=guidance_vec,
            )
            params = variables["params"]
            
            # Unbox LogicallyPartitioned parameters
            import flax.linen.spmd as flax_spmd
            params = jax.tree_util.tree_map(
                lambda x: (x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x),
                params,
                is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
            )
            params = flax.core.unfreeze(params)
            
            # 7. Convert and load PyTorch weights into JAX params
            print("Mapping and loading PyTorch weights into JAX parameters...")
            
            # Global layers
            params["txt_in"]["kernel"] = jnp.array(pt_state_dict["context_embedder.weight"].to(torch.float32).cpu().numpy().T)
            params["img_in"]["kernel"] = jnp.array(pt_state_dict["x_embedder.weight"].to(torch.float32).cpu().numpy().T)
            params["double_stream_modulation_img"]["kernel"] = jnp.array(pt_state_dict["double_stream_modulation_img.linear.weight"].to(torch.float32).cpu().numpy().T)
            params["double_stream_modulation_txt"]["kernel"] = jnp.array(pt_state_dict["double_stream_modulation_txt.linear.weight"].to(torch.float32).cpu().numpy().T)
            params["single_stream_modulation"]["kernel"] = jnp.array(pt_state_dict["single_stream_modulation.linear.weight"].to(torch.float32).cpu().numpy().T)
            
            # Double block 0
            block_idx = 0
            jax_db = params[f"double_blocks_{block_idx}"]
            prefix = f"transformer_blocks.{block_idx}."
            
            # Concatenate QKV projections
            to_q = pt_state_dict[prefix + "attn.to_q.weight"].to(torch.float32).T.cpu().numpy()
            to_k = pt_state_dict[prefix + "attn.to_k.weight"].to(torch.float32).T.cpu().numpy()
            to_v = pt_state_dict[prefix + "attn.to_v.weight"].to(torch.float32).T.cpu().numpy()
            jax_db["attn"]["i_qkv"]["kernel"] = jnp.array(np.concatenate([to_q, to_k, to_v], axis=1))
            
            add_q = pt_state_dict[prefix + "attn.add_q_proj.weight"].to(torch.float32).T.cpu().numpy()
            add_k = pt_state_dict[prefix + "attn.add_k_proj.weight"].to(torch.float32).T.cpu().numpy()
            add_v = pt_state_dict[prefix + "attn.add_v_proj.weight"].to(torch.float32).T.cpu().numpy()
            jax_db["attn"]["e_qkv"]["kernel"] = jnp.array(np.concatenate([add_q, add_k, add_v], axis=1))
            
            # Projections out
            jax_db["attn"]["i_proj"]["kernel"] = jnp.array(pt_state_dict[prefix + "attn.to_out.0.weight"].to(torch.float32).T.cpu().numpy())
            jax_db["attn"]["e_proj"]["kernel"] = jnp.array(pt_state_dict[prefix + "attn.to_add_out.weight"].to(torch.float32).T.cpu().numpy())
            
            # Norm scales
            jax_db["attn"]["query_norm"]["scale"] = jnp.array(pt_state_dict[prefix + "attn.norm_q.weight"].to(torch.float32).cpu().numpy())
            jax_db["attn"]["key_norm"]["scale"] = jnp.array(pt_state_dict[prefix + "attn.norm_k.weight"].to(torch.float32).cpu().numpy())
            jax_db["attn"]["encoder_query_norm"]["scale"] = jnp.array(pt_state_dict[prefix + "attn.norm_added_q.weight"].to(torch.float32).cpu().numpy())
            jax_db["attn"]["encoder_key_norm"]["scale"] = jnp.array(pt_state_dict[prefix + "attn.norm_added_k.weight"].to(torch.float32).cpu().numpy())
            
            # SwiGLU MLPs
            jax_db["img_mlp"]["linear_in"]["kernel"] = jnp.array(pt_state_dict[prefix + "ff.linear_in.weight"].to(torch.float32).T.cpu().numpy())
            jax_db["img_mlp"]["linear_out"]["kernel"] = jnp.array(pt_state_dict[prefix + "ff.linear_out.weight"].to(torch.float32).T.cpu().numpy())
            jax_db["txt_mlp"]["linear_in"]["kernel"] = jnp.array(pt_state_dict[prefix + "ff_context.linear_in.weight"].to(torch.float32).T.cpu().numpy())
            jax_db["txt_mlp"]["linear_out"]["kernel"] = jnp.array(pt_state_dict[prefix + "ff_context.linear_out.weight"].to(torch.float32).T.cpu().numpy())
            
            # Single block 0
            jax_sb = params[f"single_blocks_{block_idx}"]
            s_prefix = f"single_transformer_blocks.{block_idx}."
            
            # Joint projections
            jax_sb["linear1"]["kernel"] = jnp.array(pt_state_dict[s_prefix + "attn.to_qkv_mlp_proj.weight"].to(torch.float32).T.cpu().numpy())
            jax_sb["linear2"]["kernel"] = jnp.array(pt_state_dict[s_prefix + "attn.to_out.weight"].to(torch.float32).T.cpu().numpy())
            
            # Norm scales
            jax_sb["attn"]["query_norm"]["scale"] = jnp.array(pt_state_dict[s_prefix + "attn.norm_q.weight"].to(torch.float32).cpu().numpy())
            jax_sb["attn"]["key_norm"]["scale"] = jnp.array(pt_state_dict[s_prefix + "attn.norm_k.weight"].to(torch.float32).cpu().numpy())
            
            params = flax.core.freeze(params)
            
            # 8. Load inputs and run JAX block forward passes!
            print("Running mathematical parity assertions...")
            
            # A. Verify DOUBLE BLOCK 0
            # Load golden inputs for double block 0
            db_in_img = jnp.array(bundle["step_0_cond_double_block_0_input_image_latents"])
            db_in_txt = jnp.array(bundle["step_0_cond_double_block_0_input_text_latents"])
            db_in_temb_mod_img = jnp.array(bundle["step_0_cond_global_double_img_modulation_params"])
            db_in_temb_mod_txt = jnp.array(bundle["step_0_cond_global_double_txt_modulation_params"])
            
            # Generate the rotary embeddings in JAX using txt_ids and img_ids from the bundle
            txt_ids_val = jnp.array(bundle["txt_ids"])
            img_ids_val = jnp.array(bundle["img_ids"])
            ids_val = jnp.concatenate([txt_ids_val, img_ids_val], axis=1)
            db_in_rope = transformer.apply(
                {"params": params},
                ids_val,
                method=lambda self, x: self.pe_embedder(x)
            )
            
            # Run double block 0 in JAX
            db_out_img, db_out_txt = transformer.apply(
                {"params": params},
                db_in_img,
                db_in_txt,
                temb=None,
                image_rotary_emb=db_in_rope,
                temb_mod_img=db_in_temb_mod_img,
                temb_mod_txt=db_in_temb_mod_txt,
                method=lambda self, *args, **kwargs: self.double_blocks[0](*args, **kwargs)
            )
            
            # Load golden outputs (Note: PyTorch returns (text, image), so we map them correctly!)
            golden_db_out_txt = jnp.array(bundle["step_0_cond_double_block_0_output_image_latents"]) # output[0] in PT = text
            golden_db_out_img = jnp.array(bundle["step_0_cond_double_block_0_output_text_latents"])  # output[1] in PT = image
            
            # Assert parity
            np.testing.assert_allclose(
                np.array(db_out_img),
                np.array(golden_db_out_img),
                rtol=1e-2,
                atol=2.0,
                err_msg="Double block 0 image output mismatch!"
            )
            np.testing.assert_allclose(
                np.array(db_out_txt),
                np.array(golden_db_out_txt),
                rtol=1e-2,
                atol=2.0,
                err_msg="Double block 0 text output mismatch!"
            )
            print("Successfully verified JAX DoubleTransformerBlock 0 mathematical parity!")
            
            # B. Verify SINGLE BLOCK 0
            # Load golden inputs for single block 0
            sb_in = jnp.array(bundle["step_0_cond_single_block_0_input_latents"])
            sb_in_temb_mod = jnp.array(bundle["step_0_cond_global_single_joint_modulation_params"])
            
            # Run single block 0 in JAX
            sb_out = transformer.apply(
                {"params": params},
                sb_in,
                temb=None,
                image_rotary_emb=db_in_rope,
                temb_mod=sb_in_temb_mod,
                method=lambda self, *args, **kwargs: self.single_blocks[0](*args, **kwargs)
            )
            
            # Load golden outputs
            golden_sb_out = jnp.array(bundle["step_0_cond_single_block_0_output_latents"])
            
            # Assert parity
            np.testing.assert_allclose(
                np.array(sb_out),
                np.array(golden_sb_out),
                rtol=1e-2,
                atol=2.0,
                err_msg="Single block 0 output mismatch!"
            )
            print("Successfully verified JAX SingleTransformerBlock 0 mathematical parity!")

if __name__ == '__main__':
    unittest.main()
