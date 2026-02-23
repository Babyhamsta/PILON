
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from pilon_r.core.config import ModelConfig, PrimitiveConfig, MoEConfig
from pilon_r.core.model import PILONTransformer

def verify_moe():
    print("Verifying MoE functionality...")
    
    # 1. Configure MoE
    moe_config = MoEConfig(
        n_experts=4,
        top_k=2,
        load_balancing=True
    )
    
    prim_config = PrimitiveConfig(
        n_primitives=16,
        rank=16,
        top_k=4,
        moe_config=moe_config
    )
    
    model_config = ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        primitive_config=prim_config,
        ffn_type="compositional"
    )
    
    # 2. Instantiate Model
    print("Instantiating MoE model...")
    model = PILONTransformer(model_config)
    
    if not model.is_moe_model():
        print("ERROR: Model did not initialize as MoE!")
        return False
    else:
        print("Model correctly identified as MoE.")

    # 3. Create Dummy Data
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # 4. Forward Pass
    print("Running forward pass...")
    model.train() # Enable aux loss
    
    # Mock runtime override that train.py usually sets
    for layer in model.layers:
        if hasattr(layer.ffn, "runtime_top_k"):
             # For MoE FFN, runtime_top_k controls *expert* selection density
             # If None/default, it might be dense. Let's set it to top_k experts.
             layer.ffn.runtime_top_k = moe_config.top_k

    outputs = model(input_ids, labels=labels)
    
    # 5. Check Outputs
    if "loss" not in outputs:
        print("ERROR: Loss not computed.")
        return False
        
    if "aux_loss" not in outputs:
        print("ERROR: Aux loss not computed.")
        return False
        
    loss = outputs["loss"]
    aux_loss = outputs["aux_loss"]
    
    print(f"Main Loss: {loss.item():.4f}")
    print(f"Aux Loss: {aux_loss.item():.4f}")
    
    if aux_loss == 0.0:
        print("WARNING: Aux loss is exactly 0.0. This might be okay for initialized weights but suspicious.")
    
    # 6. Check Metrics
    print("Checking MoE metrics...")
    metrics = model.get_moe_metrics()
    if metrics is None:
        print("ERROR: No MoE metrics returned.")
        return False
        
    print(f"Mean Router Entropy: {metrics.get('mean_router_entropy', 'N/A')}")
    print(f"Mean Expert Similarity: {metrics.get('mean_expert_similarity', 'N/A')}")
    
    print("\n✅ MoE Verification Passed!")
    return True

if __name__ == "__main__":
    try:
        if verify_moe():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification Failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
