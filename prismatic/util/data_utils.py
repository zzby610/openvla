"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.image_processing_utils import BatchFeature


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # pixel_values = [instance["pixel_values"] for instance in instances]
        pixel_values = []
        for instance in instances:
            if isinstance(instance["pixel_values"], list):
                pixel_values.extend(instance["pixel_values"])  # 把里面的 list 展开
            else:
                pixel_values.append(instance["pixel_values"])

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    # pixel_values_dtype: torch.dtype = torch.float32
    pixel_values_dtype: torch.dtype = torch.bfloat16 #下游操作改
    

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # pixel_values = [instance["pixel_values"] for instance in instances]

        pixel_values = []
        for instance in instances:
            pv = instance["pixel_values"]
            if isinstance(pv, BatchFeature):
                pixel_values.append(pv["pixel_values"])
            elif isinstance(pv, torch.Tensor):
                pixel_values.append(pv)
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pv)}")

        pixel_values = torch.stack(pixel_values, dim=0).to(self.pixel_values_dtype)


        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]

        # print("=== DEBUG pixel_values ===")
        # print(f"type(pixel_values): {type(pixel_values)}")
        # print(f"type(pixel_values[0]): {type(pixel_values[0])}")
        # print(f"len(pixel_values): {len(pixel_values)}")
        # print(pixel_values)


        if isinstance(pixel_values, torch.Tensor):
            # 已经是Tensor了，不需要stack
            pass
        elif isinstance(pixel_values[0], torch.Tensor):
            # 是list of Tensor，需要stack
            pixel_values = torch.stack(pixel_values, dim=0)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(pixel_values))], dim=0)
                for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        

        #处理image_sizes
        # if "image_sizes" in instances[0]:
        #     image_sizes = [instance["image_sizes"] for instance in instances]
        # else:
        #     image_sizes = None
        # image_sizes_list = [instance.get("image_sizes", None) for instance in instances]


        #flatten操作从List[List[Tensor]]展平为List[Tensor]
        #shape = [1, 2] 生成时再加一个维度
        # image_sizes_list = [
        #     size.unsqueeze(0) if size.dim() == 1 else size
        #     for instance in instances
        #     if instance.get("image_sizes") is not None
        #     for size in instance["image_sizes"]
        # ]
        image_sizes_list = [torch.tensor([[1, 1]]) for _ in instances]

        # print("per-instance image_sizes:", image_sizes_list)
        # print("[DEBUG] collator pixel_values dtype:", pixel_values.dtype)



        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        if any(s is not None for s in image_sizes_list):
            output["image_sizes"] = image_sizes_list

        # print("=== DEBUG collate_fn ===")
        # print(f"batch keys: {output.keys()}")
        # print(f"batch image_sizes: {output.get('image_sizes', None)}")
        # print("image_sizes shape list:", [s.shape for s in image_sizes_list])



        return output


    # def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    #     input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
    #     pixel_values = [instance["pixel_values"] for instance in instances]
        
    #     # ✨新增：收集image_sizes
    #     if "image_sizes" in instances[0] and instances[0]["image_sizes"] is not None:
    #         image_sizes = [instance["image_sizes"] for instance in instances]
    #     else:
    #         image_sizes = None

    #     # padding
    #     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
    #     labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    #     attention_mask = input_ids.ne(self.pad_token_id)

    #     if isinstance(pixel_values[0], torch.Tensor):
    #         pixel_values = torch.stack(pixel_values, dim=0)
    #     elif isinstance(pixel_values[0], dict):
    #         pixel_values = {
    #             k: torch.stack([pixel_values[idx][k] for idx in range(len(pixel_values))], dim=0)
    #             for k in pixel_values[0]
    #         }
    #     else:
    #         raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    #     output = dict(
    #         pixel_values=pixel_values,
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #     )

    #     if image_sizes is not None:
    #         output["image_sizes"] = image_sizes

    #     return output
