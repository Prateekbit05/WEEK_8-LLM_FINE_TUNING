
# Training Report

**Generated:** 2026-03-08 12:30:15

## Configuration

- **Model:** microsoft/phi-2
- **LoRA Rank:** 16
- **LoRA Alpha:** 32
- **Epochs:** 3
- **Batch Size:** 4 (effective: 16)
- **Learning Rate:** 0.0002

## Results

| Metric | Value |
|--------|-------|
| **Training Time** | 0h 43m 14s |
| **Initial Loss** | 2.9655 |
| **Final Loss** | 1.1992 |
| **Best Eval Loss** | 1.1992 |
| **Loss Improvement** | 59.56% |
| **Trainable Params** | 7,864,320 (0.5143%) |
| **Peak GPU Memory** | 4.34 GB |
| **Adapter Size** | 30.02 MB |

## Training Samples

- Train: 1,016 samples
- Validation: 20 samples
- Total: 1,036 samples

## Model Info

- Total Parameters: 1,529,256,960
- Trainable Parameters: 7,864,320
- Frozen Parameters: 1,521,392,640

## Next Steps

1. **Test the model:** Run the generation cell below
2. **Download adapter:** Download the `adapters/` folder
3. **Use in production:** Load with PeftModel

