import torch

from typing import Optional, Type

from transformers import PreTrainedTokenizerBase

from text_generation_server.models import CausalLM
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.pb import generate_pb2


class BloomCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "CausalLMBatch":
        batch = super(BloomCausalLMBatch, cls).from_pb(
            pb=pb, tokenizer=tokenizer, device=device
        )
        batch.keys_head_dim_last = False
        return batch


class BLOOM(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
    ):
        super(BLOOM, self).__init__(
            model_id=model_id,
            revision=revision,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return BloomCausalLMBatch
