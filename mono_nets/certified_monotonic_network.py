from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


class TwoLayerPerceptron(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(in_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.fc_in(x).relu()
        out = self.fc_out(hidden)
        return hidden, out


class CertifiedMonotonicNetwork(nn.Module):
    def __init__(
        self,
        in_size_mono: int,
        in_size_non_mono: int,
        topology_mono: Sequence[Tuple[int, int]],
        topology_non_mono: Sequence[Tuple[int, int]],
        first_non_mono_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        if len(topology_mono) != len(topology_non_mono):
            raise ValueError("Lengths of `topology_mono` and `topology_non_mono` must be the same.")

        if topology_mono[-1][1] > topology_non_mono[-1][1]:
            raise ValueError(
                "The last out_size of `topology_mono` must be equal to or smaller than that of `topology_non_mono`"
            )

        self.in_size_mono = in_size_mono
        self.in_size_non_mono = in_size_non_mono
        self.topology_mono = topology_mono
        self.topology_non_mono = topology_non_mono

        self.mono_mlps = nn.ModuleList()
        self.non_mono_mlps = nn.ModuleList()

        # 最初の1つだけ、non_mono_feature を直ちに mono_feature に concat するのではなく、
        # non_mono_mlp の hidden に変換してから mono_feature に concat するため特別に処理する
        hidden_size_non_mono, out_size_non_mono = topology_non_mono[0]
        hidden_size_mono, out_size_mono = topology_mono[0]
        self.non_mono_mlps.append(TwoLayerPerceptron(in_size_non_mono, hidden_size_non_mono, out_size_non_mono))
        if first_non_mono_size is not None:
            # first_non_mono_size が指定されている場合は、first_non_mono_size に変換してから concat する
            self.compressor: nn.Module = nn.Linear(hidden_size_non_mono, first_non_mono_size)
            self.mono_mlps.append(
                TwoLayerPerceptron(in_size_mono + first_non_mono_size, hidden_size_mono, out_size_mono)
            )
        else:
            self.compressor = nn.Identity()
            self.mono_mlps.append(
                TwoLayerPerceptron(in_size_mono + hidden_size_non_mono, hidden_size_mono, out_size_mono)
            )
        in_size_mono = out_size_mono + out_size_non_mono
        in_size_non_mono = out_size_non_mono

        for (hidden_size_mono, out_size_mono), (hidden_size_non_mono, out_size_non_mono) in zip(
            topology_mono[1:], topology_non_mono[1:]
        ):
            self.non_mono_mlps.append(TwoLayerPerceptron(in_size_non_mono, hidden_size_non_mono, out_size_non_mono))
            self.mono_mlps.append(TwoLayerPerceptron(in_size_mono, hidden_size_mono, out_size_mono))
            in_size_mono = out_size_mono + out_size_non_mono
            in_size_non_mono = out_size_non_mono

    def forward(self, mono_feature: Tensor, non_mono_feature: Tensor) -> Tensor:
        mono_mlp, non_mono_mlp = self.mono_mlps[0], self.non_mono_mlps[0]
        hidden, non_mono_feature = non_mono_mlp(non_mono_feature)
        mono_feature = mono_mlp(torch.cat([mono_feature, self.compressor(hidden)], dim=-1).clamp(0, 1))[1]

        for mono_mlp, non_mono_mlp in zip(self.mono_mlps[1:], self.non_mono_mlps[1:]):
            mono_feature = mono_mlp(torch.cat([mono_feature, non_mono_feature], dim=-1).clamp(0, 1))
            non_mono_feature = non_mono_mlp(non_mono_feature)

        out = non_mono_feature.clone()
        out[..., : mono_feature.shape[-1]] += mono_feature
        return out

    def certify(self) -> bool:
        pass
