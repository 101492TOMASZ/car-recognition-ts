import torch
import pytest

def test_load_model_weights():
    model_path = 'model/car_model2.pth'
    device = 'cpu'
    
    # Mocking the model loading process
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        assert checkpoint is not None
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")