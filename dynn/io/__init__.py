# Initializes the io sub-package for input/output and environment interaction.

from .input_encoders import (
    BaseInputEncoder,
    GaussianEncoder,
    DirectCurrentInjector
)

from .output_decoders import (
    BaseOutputDecoder,
    InstantaneousSpikeCountDecoder,
    BidirectionalThresholdDecoder
)

from .reward_processors import (
    BaseRewardProcessor,
    SlidingWindowSmoother
)

__all__ = [
    # Input Encoders
    'BaseInputEncoder',
    'GaussianEncoder',
    'DirectCurrentInjector',
    # Output Decoders
    'BaseOutputDecoder',
    'InstantaneousSpikeCountDecoder',
    'BidirectionalThresholdDecoder',
    # Reward Processors
    'BaseRewardProcessor',
    'SlidingWindowSmoother'
]

# Placeholder for I/O and environment interaction components 