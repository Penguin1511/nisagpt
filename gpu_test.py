import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device='cuda')   # GPU'da rastgele bir matris
    y = x @ x                                    # Matris çarpımı
    print("GPU'da matris çarpımı çalıştı, sonucun şekli:", y.shape)
    