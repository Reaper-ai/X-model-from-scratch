"""
Test script to verify nn.Module JEPA and generate torchviz architecture graph.
"""

import torch
from torchviz import make_dot

from modules import JEPAModel


def test_module_jepa():
    """Test the nn.Module JEPA implementation."""
    torch.manual_seed(0)

    # Create model
    model = JEPAModel(
        obs_channels=1,
        room_size=8,
        latent_dim=32,
        action_dim=4,
        hidden_dim=64,
        tau=0.99
    )

    print("Model created:")
    print(f"  Encoder params: {sum(p.numel() for p in model.encoder.parameters())}")
    print(f"  Predictor params: {sum(p.numel() for p in model.predictor.parameters())}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, 1, 8, 8)
    actions = torch.randint(0, 4, (batch_size,))
    next_obs = torch.randn(batch_size, 1, 8, 8)

    outputs = model(obs, actions, next_obs)
    print(f"\nForward pass:")
    print(f"  embeddings: {outputs['embeddings'].shape}")
    print(f"  target: {outputs['target'].shape}")
    print(f"  predicted: {outputs['predicted'].shape}")

    # Test loss
    loss = model.jepa_loss(outputs['predicted'], outputs['target'], outputs['embeddings'])
    print(f"  JEPA loss: {loss.item():.4f}")

    # Test collapse metric
    collapse = model.collapse_metric(outputs['embeddings'])
    print(f"  Collapse metric: {collapse.item():.4f}")

    # Test EMA update
    model.update_target()
    print("  EMA target updated")

    # Test rollout
    initial = outputs['embeddings'][0]
    action_seq = torch.randint(0, 4, (5,))
    rollout = model.rollout(initial, action_seq)
    print(f"  Rollout shape: {rollout.shape}")

    # Test LinearProbe
    from modules import LinearProbe
    probe = LinearProbe(latent_dim=32, state_dim=2)
    states = probe(outputs['embeddings'])
    print(f"  Probe output: {states.shape}")

    print("\nAll tests passed!")
    return model, obs, actions, next_obs


def generate_torchviz_graph():
    """Generate architecture graph using torchviz."""
    model, obs, actions, next_obs = test_module_jepa()

    # Get outputs for graph
    outputs = model(obs, actions, next_obs)
    loss = model.jepa_loss(outputs['predicted'], outputs['target'], outputs['embeddings'])

    # Generate graph
    print("\nGenerating torchviz graph...")
    dot = make_dot(loss, params=dict(model.named_parameters()))
    dot.format = 'svg'
    dot.render('images/jepa_torchviz', cleanup=True)
    print("Graph saved to images/jepa_torchviz.svg")


if __name__ == "__main__":
    test_module_jepa()
    try:
        generate_torchviz_graph()
    except Exception as e:
        print(f"torchviz not available or failed: {e}")
        print("Install with: pip install torchviz")