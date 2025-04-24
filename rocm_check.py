import torch

print("Czy ROCm jest dostępny:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nazwa urządzenia:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Brak dostępnych urządzeń ROCm/CUDA.")