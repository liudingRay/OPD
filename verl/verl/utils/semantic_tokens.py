# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared formatting-only token filter for EU analysis and routing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


SEMANTIC_LATEX_SCAFFOLDING = {
    "begin",
    "big",
    "bigg",
    "boxed",
    "displaystyle",
    "end",
    "frac",
    "left",
    "mathbf",
    "mathrm",
    "right",
    "text",
}
SEMANTIC_MATH_SYMBOLS = frozenset("+-−×÷*/=<>≤≥±∑√^")


def readable_token_piece(token_piece: str) -> str:
    """Convert common tokenizer whitespace markers to their readable form."""
    return token_piece.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def is_semantic_token(decoded_piece: str) -> bool:
    """Keep content-bearing pieces while dropping Markdown/LaTeX scaffolding."""
    piece = readable_token_piece(decoded_piece).strip()
    if not piece:
        return False
    normalized = piece.strip("*_`#")
    if not normalized or normalized in {"$", "$$", "\\", "{", "}", "}{", "[", "]", "(", ")"}:
        return False
    command = normalized.lstrip("\\")
    if command in SEMANTIC_LATEX_SCAFFOLDING:
        return False
    if len(normalized) > 1 and set(normalized) <= set("-_*`#"):
        return False
    return any(character.isalnum() for character in normalized) or any(
        character in SEMANTIC_MATH_SYMBOLS for character in normalized
    )


def semantic_token_mask(tokenizer, token_ids: torch.Tensor | Sequence[Sequence[int]]) -> torch.Tensor:
    """Return a CPU ``[B,T]`` mask using the same predicate as EU reports.

    The mask classifies token positions only. Wordpiece/math span expansion in
    the report remains presentation-only and must not change the routing count.
    """
    import torch

    if torch.is_tensor(token_ids):
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token IDs with shape [B,T], got {tuple(token_ids.shape)}")
        ids = token_ids.detach().cpu().long()
    else:
        rows = [list(row) for row in token_ids]
        if rows and any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Token ID rows must have equal lengths")
        ids = torch.as_tensor(rows, dtype=torch.long)
    if ids.ndim != 2:
        raise ValueError(f"Expected token IDs with shape [B,T], got {tuple(ids.shape)}")
    if ids.numel() == 0:
        return torch.zeros_like(ids, dtype=torch.bool)

    unique_ids, inverse = torch.unique(ids.reshape(-1), sorted=False, return_inverse=True)
    pieces = tokenizer.convert_ids_to_tokens(unique_ids.tolist())
    if len(pieces) != len(unique_ids):
        raise ValueError("Tokenizer returned a different number of token pieces than token IDs")
    unique_mask = torch.tensor([is_semantic_token(piece) for piece in pieces], dtype=torch.bool)
    return unique_mask[inverse].reshape_as(ids)
