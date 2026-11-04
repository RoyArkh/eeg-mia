"""Target models. Each file in this folder builds one model.

Every model file has a function

    build(n_channels, n_times, sampling_rate, dropout) -> torch.nn.Module

that returns a network mapping [batch, channels, time] to one logit per window (binary
task, trained with binary cross-entropy). The settings are those of Cobilean et al. (2025)
(github.com/CobV/MIA_BCI) that took effect in mia_bci/exp1_models.py.

To add a model, write models/<name>.py with a build() function and add it to MODELS below.
"""

from models import conv1d, eeg_inception, eegnet, eegnet2tl, shallow_convnet, test_network, tidnet

MODELS = {
    # the six CNNs of Cobilean et al. Experiment 1 and the HEALTHINF paper
    "EEGNet": eegnet,
    "ShallowConvNet": shallow_convnet,
    "Conv1DModel": conv1d,
    "EEGNet2TL": eegnet2tl,
    "TIDNet": tidnet,
    "EEGInception": eeg_inception,
    # the size-controllable CNN of Cobilean et al. Experiment 4
    "TestNetwork": test_network,
}


def build_model(name, n_channels, n_times, sampling_rate, dropout, **options):
    """Build model `name`. `options` go to its build() function (e.g. TestNetwork's size)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(MODELS)}")
    return MODELS[name].build(n_channels, n_times, sampling_rate, dropout, **options)


def resolve_names(names):
    """Check model names from the command line. "all" means every model."""
    if "all" in names:
        return list(MODELS)
    unknown = [name for name in names if name not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Available: {', '.join(MODELS)}, all")
    return list(names)


def forward_logits(model, x):
    """One logit per window, shape [batch]. Fails if a model returns anything else."""
    return model(x).reshape(len(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
