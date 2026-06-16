import unittest
import numpy as np
from maxdiffusion.generate_flux2klein import load_or_generate_latents

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

if __name__ == '__main__':
    unittest.main()
