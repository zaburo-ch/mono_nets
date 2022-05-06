import itertools
from typing import List, Optional, Sequence, Tuple

import mip
import numpy as np
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


def find_monotonicity_counterexample(
    mlp: TwoLayerPerceptron, mono_feature_num: int, verbose: bool
) -> Optional[Tuple[int, int, Tensor]]:
    w1 = mlp.fc_in.weight.detach().cpu().numpy().astype(np.float64)
    b1 = mlp.fc_in.bias.detach().cpu().numpy().astype(np.float64)
    w2 = mlp.fc_out.weight.detach().cpu().numpy().astype(np.float64)

    hidden_size, in_size = w1.shape
    out_size = w2.shape[0]

    # for feature_index in range(mono_feature_num):
    for out_index, feature_index in itertools.product(range(out_size), range(mono_feature_num)):
        m = mip.Model()
        m.verbose = verbose

        # z: 中間層の ReLU が activate されているかどうか
        z = m.add_var_tensor((hidden_size,), "z", var_type=mip.BINARY)
        # x: [0, 1] に正規化された入力
        x = m.add_var_tensor((in_size,), "x", lb=0.0, ub=1.0, var_type=mip.CONTINUOUS)

        m.objective = (w2[out_index, :] * w1[:, feature_index]) @ z

        upper = np.sum(np.maximum(w1, 0.0), axis=1) + b1  # (hidden_size,)
        lower = np.sum(np.minimum(w1, 0.0), axis=1) + b1  # (hidden_size,)

        # mip は float64 で計算するが torch は float32 で計算するので、
        # float32 にしても z が一致するように 0 周りに eps だけマージンをとる
        eps = 1e-6

        # wx + b_i <= u_i z_i - eps * (1 - z_i)
        m += (w1 @ x) + (-(upper + eps) * z) <= -b1 - eps

        # wx + b_i >= l_i (1 - z_i) + eps * z_i
        m += (-w1 @ x) + ((-lower + eps) * z) <= b1 - lower

        # こっちが論文の制約だが、これだと float32 で計算した時に z が変化し grad の符号が変化することがある
        # # wx + b_i <= u_i z_i
        # m += (w1 @ x) + (-upper * z) <= -b1

        # # wx + b_i >= l_i (1 - z_i)
        # m += (-w1 @ x) + (-lower * z) <= b1 - lower

        m.optimize()

        # print(f"Objective: {m.objective_value}, out_weight: {out_weight}")
        if verbose:
            print(f"(out_index, feature_index): {(out_index, feature_index)} Objective: {m.objective_value}")

        if m.objective_value < 0.0:
            x_tensor = torch.tensor([x_i.x for x_i in x])
            x_tensor.requires_grad = True
            (grad,) = torch.autograd.grad(mlp(x_tensor)[1][feature_index], x_tensor)
            grad = grad[feature_index]
            assert grad.item() < 0.0
            return (out_index, feature_index, x_tensor.detach())

    return None


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

        self.mono_mlps: List[TwoLayerPerceptron] = nn.ModuleList()
        self.non_mono_mlps: List[TwoLayerPerceptron] = nn.ModuleList()

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
            mono_feature = mono_mlp(torch.cat([mono_feature, non_mono_feature], dim=-1).clamp(0, 1))[1]
            non_mono_feature = non_mono_mlp(non_mono_feature)[1]

        out = non_mono_feature.clone()
        out[..., : mono_feature.shape[-1]] += mono_feature
        return out

    def certify(self, verbose: bool = True) -> bool:
        in_size_mono = self.in_size_mono
        for mlp in self.mono_mlps:
            ce = find_monotonicity_counterexample(mlp, in_size_mono, verbose)
            if ce is not None:
                return False
            in_size_mono = mlp.fc_out.out_features
        return True

    # def sample_regularizer(self, batch_size: int, offset: float = 0.2) -> Tensor:
    #     in_size_mono = self.in_size_mono
    #     device = next(iter(self.parameters())).device

    #     R = 0.0
    #     for mlp in self.mono_mlps:
    #         mono_feature = torch.rand((batch_size, in_size_mono), device=device)
    #         mono_feature.requires_grad = True
    #         non_mono_feature = torch.rand((batch_size, mlp.fc_in.in_features - in_size_mono), device=device)
    #         out = mlp(torch.cat([mono_feature, non_mono_feature], dim=1))[1]
    #         for out_index in range(out.shape[1]):
    #             (grad,) = torch.autograd.grad(
    #                 out[:, out_index].sum(), mono_feature, create_graph=True, retain_graph=True
    #             )
    #             R += ((offset + -grad).relu() ** 2).mean() / out.shape[1]

    #         in_size_mono = out.shape[1]

    #     return R

    def sample_regularizer(self, batch_size: int, offset: float = 0.2) -> Tensor:
        # 著者実装が mean じゃなくて max をとっているので真似てみる
        # https://github.com/gnobitab/CertifiedMonotonicNetwork/blob/main/compas/train.py#L60-L75
        in_size_mono = self.in_size_mono
        device = next(iter(self.parameters())).device

        R = torch.tensor(-np.inf, device=device)
        for mlp in self.mono_mlps:
            mono_feature = torch.rand((batch_size, in_size_mono), device=device)
            mono_feature.requires_grad = True
            non_mono_feature = torch.rand((batch_size, mlp.fc_in.in_features - in_size_mono), device=device)
            out = mlp(torch.cat([mono_feature, non_mono_feature], dim=1))[1]
            for out_index in range(out.shape[1]):
                (grad,) = torch.autograd.grad(
                    out[:, out_index].sum(), mono_feature, create_graph=True, retain_graph=True
                )
                R_i = torch.max((offset + -grad).relu() ** 2)
                if R < R_i:
                    R = R_i
            in_size_mono = out.shape[1]

        return R


if __name__ == "__main__":
    # 自明に Monotonic な例で certify がちゃんと True を返すことを確認
    model = CertifiedMonotonicNetwork(4, 2, [(100, 10), (100, 1)], [(100, 10), (100, 1)])
    with torch.no_grad():
        for mlp in model.mono_mlps:
            mlp.fc_in.weight.data.clamp_(min=0)
            mlp.fc_out.weight.data.clamp_(min=0)
    print(model.certify())  # OK

    # Regularizer で Monotonic にできることを確認
    model = CertifiedMonotonicNetwork(4, 2, [(100, 10), (100, 1)], [(100, 10), (100, 1)])
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for _ in range(10 ** 4):  # 10 ** 5 でもダメだったよ...
        R = model.sample_regularizer(1024)
        print(R, end="\r")
        loss = 1e4 * R
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print()
    print(model.certify())  # ダメ...
