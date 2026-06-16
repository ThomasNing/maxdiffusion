# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
import numpy as np
import os

def load_or_generate_latents(config):
    """
    Loads saved latents if use_latents is True, otherwise generates random latents.
    """
    batch_size = config.get("batch_size", 1)
    height = config.get("height", 512)
    width = config.get("width", 512)
    use_latents = config.get("use_latents", False)

    # Flux latents typically have 32 channels and are downsampled by 8
    num_channels_latents = 32
    latent_height = height // 8
    latent_width = width // 8
    latent_shape = (batch_size, num_channels_latents, latent_height, latent_width)

    if use_latents:
        print("use_latents is True. Loading latents from disk...")
        bundle_path = "flux2_klein_complete_diagnostic_bundle.npz"
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Expected to find {bundle_path} but it was not found.")
        
        bundle = np.load(bundle_path)
        if "initial_pipeline_latents" in bundle:
            latents = bundle["initial_pipeline_latents"]
            print(f"Successfully loaded initial_pipeline_latents with shape: {latents.shape}")
            
            # Ensure shape matches what we expect
            if latents.shape != latent_shape:
                print(f"Warning: Loaded latent shape {latents.shape} does not match expected shape {latent_shape}.")
        else:
            raise KeyError(f"'initial_pipeline_latents' not found in {bundle_path}")
    else:
        print(f"use_latents is False. Generating random gaussian noise with shape: {latent_shape}...")
        # Fix seed for reproducibility in testing
        np.random.seed(42)  
        latents = np.random.randn(*latent_shape).astype(np.float32)

    return latents


def main(argv):
    # Temporary mock config for testing execution without full pyconfig setup yet
    config = {
        "use_latents": True,
        "batch_size": 1,
        "height": 512,
        "width": 512,
    }
    
    latents = load_or_generate_latents(config)
    print(f"Final latents shape ready for pipeline: {latents.shape}")

if __name__ == "__main__":
    app.run(main)
