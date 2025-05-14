"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

        # self.action_token_begin_idx: int = len(tokenizer) - self.n_bins
        # self.action_token_end_idx: int = len(tokenizer) - 1

        # self.action_token_begin_idx = 128000  #？
        #[128000-256,128000]or[128000,128000+256]?


    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # token_ids = self.action_token_begin_idx + (discretized_action - 1)

        # if len(token_ids.shape) == 1:
        #     return self.tokenizer.decode(token_ids.tolist())
        # else:
        #     return self.tokenizer.batch_decode(token_ids.tolist())

        # Handle single element vs. batch
        # if len(discretized_action.shape) == 1:
        #     return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        # else:
        #     return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())
        
            # Subtract 1 to make bin index start from 0

        action_indices = (discretized_action - 1).clip(0, self.n_bins - 1)

        if action_indices.ndim == 1:
            return [f"<action_{i}>" for i in action_indices]
        else:
            return [[f"<action_{i}>" for i in row] for row in action_indices]
        

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]
    

    # def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    #     action_token_ids = np.asarray(action_token_ids)
    #     flat_ids = action_token_ids.flatten()
    #     decoded = np.full(flat_ids.shape, fill_value=np.nan, dtype=np.float32)

    #     valid = (flat_ids >= self.action_token_begin_idx) & (flat_ids <= self.action_token_end_idx)
    #     bin_idx = flat_ids[valid] - self.action_token_begin_idx
    #     bin_idx = np.clip(bin_idx, 0, self.bin_centers.shape[0] - 1)
    #     decoded[valid] = self.bin_centers[bin_idx]

    #     return decoded.reshape(action_token_ids.shape)

        

    @property
    def vocab_size(self) -> int:
        return self.n_bins
