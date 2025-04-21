import torch
from prune import get_ranks, get_pruned_model

# a tiny toy model matching the net_1 / net_2 API:
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        # flatten + linear to 10 classes
        self.net_2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 32 * 32, 10),
        )
    def forward(self, x):
        x = self.net_1(x)
        return self.net_2(x)

def main():
    model = ToyModel().to('mps')
    # fake data loader: 5 batches of random inputs & labels
    batches = [ (torch.randint(0,10,(4,)), torch.randn(4,3,32,32)) for _ in range(5) ]
    # monkey-patch train data
    import prune    
    prune.get_train_data = lambda **kw: batches
    
    ranks = get_ranks(model)
    assert len(ranks) > 0, "ranks should not be empty"
    total_before = sum(m.out_channels for m in model.net_1 if isinstance(m, torch.nn.Conv2d))
    # record parameter count before pruning
    num_params_before = sum(p.numel() for p in model.parameters())
    
    # perform pruning
    pruned = get_pruned_model(ranks=ranks, model=model, pruning_amount=0.25)
    
    total_after = sum(m.out_channels for m in pruned.net_1 if isinstance(m, torch.nn.Conv2d))
    print(f"Filters before: {total_before}, after: {total_after}")
    assert total_after < total_before, "pruning did not reduce filters"
    
    # check parameter count decreased
    num_params_after = sum(p.numel() for p in pruned.parameters())
    print(f"Parameters before: {num_params_before}, after: {num_params_after}")
    assert num_params_after < num_params_before, "pruning did not reduce parameters"
    # check if the model can still run
    # this is a bit tricky, as the model size is not directly accessible
    # but we can check if the model can still run
    # by running a forward pass
    # with a random input
    input = torch.randn(1, 3, 32, 32).to('mps')
    output = pruned(input)
    assert output is not None, "model did not run"

if __name__ == "__main__":
    main()
    print("Smoke test passed!")
