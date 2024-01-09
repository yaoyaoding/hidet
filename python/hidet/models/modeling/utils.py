import tqdm


def copy_weights(torch_model, hidet_model):
    import hidet

    found_tensors = []
    for name, tensor in tqdm(list(torch_model.named_parameters()), desc='copying weights'):
        mod = hidet_model
        for m_name in name.split('.'):
            mod = getattr(mod, m_name)

        if not isinstance(mod, hidet.Tensor):
            print(type(mod))
            raise ValueError(f"hidet/hf mismatch at {name}")

        src = hidet.from_torch(tensor).to(mod.dtype, mod.device)
        if len(src.shape) != len(mod.shape) or any(a != b for a, b in zip(src.shape, mod.shape)):
            raise RuntimeError(f"hidet/hf shape mismatch at {name}, hidet: {mod.shape}, torch: {src.shape}")
        found_tensors.append(mod)
        mod.copy_(src)

    buffer_names = set(name for name, _ in torch_model.named_buffers())

    for name, tensor in hidet_model.named_parameters():
        if tensor not in found_tensors and name not in buffer_names:
            raise ValueError(f'{name} not copied')

