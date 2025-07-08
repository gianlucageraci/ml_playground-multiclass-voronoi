from torch import nn


class Net(nn.Module):
    """Simple Neural Net for Mulitclass Classification"""

    def __init__(self, input_size: int, n_classes: int):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Embedding(input_size, 2),
            # Add Activation Function?
            nn.Linear(2, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Forward pass the input from the linear layer stack consisting
        of an Embedding that transforms the input into a 2-dim vector,
        which is passed on to a Linear Layer + SOftmax that do the
        classification into n_classes.
        """
        logits = self.linear_stack(x)
        return logits
