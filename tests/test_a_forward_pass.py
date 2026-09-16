import torch

from transformer.transformer import Transformer


def test_transformer_forward_pass():
    """Verify that a full forward pass executes without errors and outputs correct dimensions."""
    # 1. Define hyperparameters
    src_vocab_size = 1000  
    tgt_vocab_size = 1000  
    d_model = 512          
    num_heads = 8          
    num_layers = 6         
    d_ff = 2048            
    dropout = 0.1          

    # 2. Instantiate the Transformer model
    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout
    )

    # 3. Setup Example Input Tensors
    batch_size = 64
    src_seq_len = 50
    tgt_seq_len = 60

    src_data = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    src_data[:, src_seq_len - 5:] = 0  # Simulate padding at the end

    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    tgt_data[:, 0] = 1                 # Start token
    tgt_data[:, tgt_seq_len - 5:] = 0  # Simulate padding at the end

    # 4. Execute Forward pass
    output = model(src_data, tgt_data)

    # 5. Assert that the mathematical output dimensions match the decoder target space
    expected_shape = (batch_size, tgt_seq_len, tgt_vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
