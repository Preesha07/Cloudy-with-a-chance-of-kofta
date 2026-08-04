"""
Linear-probe diagnostic: how much SAR information is in netH's activations?

The premise of the hallucination student (Hoffman et al. 2016) is that netH,
which only ever sees cloudy optical, learns an internal representation that
stands in for the SAR the teacher had. This script tests that premise directly
instead of inferring it from downstream PSNR.

For each hooked UNet block it fits a ridge-regression linear probe

    real_C  ~  W · activations + b

on a train split of tiles and reports held-out R^2 on a test split. Four
feature sources are probed under an identical split, identical lambda grid,
and identical solver, so the numbers are comparable:

  netH_trained   the student's trained hallucination module, on real_A
  netH_random    the same architecture, randomly initialised, on real_A
                 -> THE FLOOR. Random nonlinear features of the optical input.
                    If netH_trained does not beat this, training put no SAR
                    information into netH that was not already trivially
                    available from the optical input.
  teacher_G2     the teacher's netG2, on real_C
                 -> sanity check. It is fed the SAR itself, so R^2 should be
                    high; if it is not, the probe is broken, not the model.
  raw_optical    the 3 raw optical channels, average-pooled to block resolution
                 -> what a linear read of the input alone gives. Low-dimensional
                    (D=3), so it is a reference point, not a fair floor;
                    netH_random is the fair floor.

It also reports val-split L_hall (Hoffman's sigmoid-L2, the exact training
objective) for trained vs random netH -- the distillation *fidelity* number in
the sense of Stanton et al. (NeurIPS 2021): did the student actually come to
match the teacher on held-out data, separately from whether that helped.

All statistics are accumulated in a single pass. SSE for any lambda is
recovered in closed form from (X^T X, X^T y, sum y^2) per split, so no
features are ever stored and no split is visited twice.

Usage (from khushi/student/):
    python probe_sar.py \
        --dataroot ../../SEN12MSCR_student \
        --checkpoints_dir ./outputs/checkpoints \
        --name sen12mscr_student_v1 \
        --teacher_name sen12mscr_teacher_v1 \
        --phase val --which_epoch 20
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import networks
from models.pix2pix_attn_student_model import _get_unet_blocks


# ── Options ───────────────────────────────────────────────────────────────────

class ProbeOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument('--teacher_checkpoints_dir', type=str, default='./outputs/checkpoints')
        self.parser.add_argument('--teacher_name', type=str, default='sen12mscr_teacher_v1')
        self.parser.add_argument('--teacher_which_epoch', type=str, default='latest')
        self.parser.add_argument('--hook_layers', type=str, default='3,4,5,6,7',
                                 help='comma-separated UNet block indices, 0 = outermost')
        self.parser.add_argument('--probe_seed', type=int, default=0,
                                 help='seed for the tile train/select/test split and for netH_random')
        self.parser.add_argument('--split', type=float, nargs=3, default=[0.6, 0.15, 0.25],
                                 metavar=('FIT', 'SELECT', 'TEST'),
                                 help='tile fractions for probe fit / lambda selection / reporting')
        self.parser.add_argument('--lambdas', type=float, nargs='+',
                                 default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4],
                                 help='ridge penalty grid; selected on the SELECT split')
        self.parser.add_argument('--out_csv', type=str, default='probe_sar_results.csv')


# ── Streaming sufficient statistics for ridge regression ──────────────────────

class Accum:
    """Accumulates (X^T X, X^T y, sum y^2, sum y, n) for a feature block.

    X is the activation matrix with a bias column appended, y is the SAR target.
    Products are formed in float32 on GPU and summed into float64, so the
    accumulator does not drift over ~10^5-10^6 rows.
    """

    def __init__(self, dim, device, n_targets=3):
        self.d = dim
        self.A = torch.zeros(dim + 1, dim + 1, dtype=torch.float64, device=device)
        self.b = torch.zeros(dim + 1, n_targets, dtype=torch.float64, device=device)
        self.syy = torch.zeros(n_targets, dtype=torch.float64, device=device)
        self.sy = torch.zeros(n_targets, dtype=torch.float64, device=device)
        self.n = 0

    def add(self, X, y):
        ones = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
        Xb = torch.cat([X, ones], dim=1)
        self.A += (Xb.t() @ Xb).double()
        self.b += (Xb.t() @ y).double()
        self.syy += (y * y).sum(0).double()
        self.sy += y.sum(0).double()
        self.n += X.shape[0]

    def solve(self, lam):
        """Ridge solution. The bias column is not penalised."""
        reg = torch.eye(self.d + 1, dtype=torch.float64, device=self.A.device) * lam
        reg[self.d, self.d] = 0.0
        return torch.linalg.solve(self.A + reg, self.b)

    def sse(self, W):
        """Per-target sum of squared errors of W on this split, in closed form."""
        return (torch.diagonal(W.t() @ self.A @ W)
                - 2.0 * torch.diagonal(W.t() @ self.b)
                + self.syy)

    def sst(self, y_mean):
        """Per-target total sum of squares about a constant predictor y_mean."""
        return self.syy - 2.0 * y_mean * self.sy + self.n * y_mean ** 2


# ── Activation capture ────────────────────────────────────────────────────────

def register(net, idxs, store):
    blocks = _get_unet_blocks(net)
    handles = []
    for i in idxs:
        if i >= len(blocks):
            raise IndexError('hook layer %d >= block count %d' % (i, len(blocks)))

        def make(i):
            def fn(module, inp, out):
                store[i] = out.detach()
            return fn
        handles.append(blocks[i].register_forward_hook(make(i)))
    return handles


def main():
    opt = ProbeOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    idxs = [int(s) for s in opt.hook_layers.split(',') if s.strip() != '']
    device = torch.device('cuda:%d' % opt.gpu_ids[0]) if opt.gpu_ids else torch.device('cpu')

    # ── Networks ─────────────────────────────────────────────────────────────
    def build():
        return networks.define_G2(opt.input_nc, opt.output_nc, opt.ngf,
                                  opt.which_model_netG2, opt.norm,
                                  not opt.no_dropout, opt.init_type, opt.gpu_ids)

    student_dir = os.path.join(opt.checkpoints_dir, opt.name)
    teacher_dir = os.path.join(opt.teacher_checkpoints_dir, opt.teacher_name)

    netH = build()
    h_path = os.path.join(student_dir, '%s_net_H.pth' % opt.which_epoch)
    print('Loading netH  from %s' % h_path)
    netH.load_state_dict(torch.load(h_path, map_location=device), strict=False)

    torch.manual_seed(opt.probe_seed)
    netH_rand = build()  # define_G2 re-inits weights; seeded above for reproducibility

    netG2_t = build()
    g2_path = os.path.join(teacher_dir, '%s_net_G2.pth' % opt.teacher_which_epoch)
    print('Loading netG2 from %s' % g2_path)
    netG2_t.load_state_dict(torch.load(g2_path, map_location=device), strict=False)

    for net in (netH, netH_rand, netG2_t):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    acts = {'netH_trained': {}, 'netH_random': {}, 'teacher_G2': {}}
    handles = (register(netH, idxs, acts['netH_trained'])
               + register(netH_rand, idxs, acts['netH_random'])
               + register(netG2_t, idxs, acts['teacher_G2']))

    # ── Data + tile split ────────────────────────────────────────────────────
    dataset = CreateDataLoader(opt).load_data()
    n_tiles = min(len(dataset), opt.how_many)
    rng = np.random.RandomState(opt.probe_seed)
    perm = rng.permutation(n_tiles)
    f_fit, f_sel, _ = opt.split
    n_fit = int(round(f_fit * n_tiles))
    n_sel = int(round(f_sel * n_tiles))
    split_of = np.empty(n_tiles, dtype=object)
    split_of[perm[:n_fit]] = 'fit'
    split_of[perm[n_fit:n_fit + n_sel]] = 'sel'
    split_of[perm[n_fit + n_sel:]] = 'test'
    print('\nTiles: %d  (fit %d / select %d / test %d)  seed=%d'
          % (n_tiles, n_fit, n_sel, n_tiles - n_fit - n_sel, opt.probe_seed))
    print('Split is a seeded random permutation of tiles. Tiles from one ROI '
          'overlap spatially, so absolute R^2 is optimistic -- but the split is\n'
          'identical for every feature source, so the COMPARISON between sources '
          'is the part to read.\n')

    sources = ['netH_trained', 'netH_random', 'teacher_G2', 'raw_optical']
    accums = {}          # (source, layer, split) -> Accum
    hall = {'netH_trained': [], 'netH_random': []}   # val L_hall, per test tile

    # ── Single pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= n_tiles:
                break
            real_A = data['A'].to(device)
            real_C = data['C'].to(device)

            netH(real_A)
            netH_rand(real_A)
            netG2_t(real_C)

            sp = split_of[i]
            for layer in idxs:
                h, w = acts['netH_trained'][layer].shape[-2:]
                y = F.adaptive_avg_pool2d(real_C, (h, w))[0].permute(1, 2, 0).reshape(-1, 3)
                for src in sources:
                    if src == 'raw_optical':
                        feat = F.adaptive_avg_pool2d(real_A, (h, w))
                    else:
                        feat = acts[src][layer]
                    X = feat[0].permute(1, 2, 0).reshape(-1, feat.shape[1])
                    key = (src, layer, sp)
                    if key not in accums:
                        accums[key] = Accum(X.shape[1], device)
                    accums[key].add(X, y)

            if sp == 'test':
                for src in ('netH_trained', 'netH_random'):
                    v = sum(torch.mean((torch.sigmoid(acts['teacher_G2'][l])
                                        - torch.sigmoid(acts[src][l])) ** 2).item()
                            for l in idxs)
                    hall[src].append(v)

            if (i + 1) % 100 == 0:
                print('  ... %d/%d tiles' % (i + 1, n_tiles))

    for hd in handles:
        hd.remove()

    # ── Solve, select lambda on 'sel', report on 'test' ──────────────────────
    rows = []
    for layer in idxs:
        for src in sources:
            a_fit = accums[(src, layer, 'fit')]
            a_sel = accums[(src, layer, 'sel')]
            a_test = accums[(src, layer, 'test')]
            y_mean = a_fit.b[a_fit.d] / a_fit.n           # train mean of the target

            best = None
            for lam in opt.lambdas:
                W = a_fit.solve(lam)
                r2_sel = 1.0 - (a_sel.sse(W).sum() / a_sel.sst(y_mean).sum()).item()
                if best is None or r2_sel > best[0]:
                    best = (r2_sel, lam, W)
            _, lam, W = best

            sse = a_test.sse(W)
            sst = a_test.sst(y_mean)
            r2_all = 1.0 - (sse.sum() / sst.sum()).item()
            r2_ch = (1.0 - sse / sst).cpu().numpy()
            rows.append(dict(layer=layer, source=src, dim=a_fit.d, rows_fit=a_fit.n,
                             lam=lam, r2=r2_all,
                             r2_vv=r2_ch[0], r2_vh=r2_ch[1], r2_diff=r2_ch[2]))

    # ── Report ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 88)
    print('LINEAR PROBE: predict real_C (SAR) from activations. Held-out R^2, higher = more')
    print('SAR information linearly present. R^2 <= 0 means no better than a constant.')
    print('=' * 88)
    print('%-6s %-14s %6s %9s %8s %8s %8s %8s %8s'
          % ('layer', 'source', 'dim', 'rows_fit', 'lambda', 'R2', 'R2_VV', 'R2_VH', 'R2_diff'))
    for layer in idxs:
        for r in [x for x in rows if x['layer'] == layer]:
            print('%-6d %-14s %6d %9d %8.3g %8.4f %8.4f %8.4f %8.4f'
                  % (r['layer'], r['source'], r['dim'], r['rows_fit'], r['lam'],
                     r['r2'], r['r2_vv'], r['r2_vh'], r['r2_diff']))
        print('-' * 88)

    print('\nKEY COMPARISON  netH_trained - netH_random  (the training effect):')
    for layer in idxs:
        t = next(x for x in rows if x['layer'] == layer and x['source'] == 'netH_trained')
        rd = next(x for x in rows if x['layer'] == layer and x['source'] == 'netH_random')
        print('  layer %d:  %+.4f   (trained %.4f  vs  random %.4f)'
              % (layer, t['r2'] - rd['r2'], t['r2'], rd['r2']))

    for layer in idxs:
        n_rows = next(x for x in rows if x['layer'] == layer)['rows_fit']
        dim = next(x for x in rows if x['layer'] == layer
                   and x['source'] == 'netH_trained')['dim']
        if n_rows < 10 * dim:
            print('\nWARNING layer %d: %d fit rows for %d features (<10x). Ridge keeps this '
                  'solvable\n  but the probe is near-interpolating; treat that row as weak evidence.'
                  % (layer, n_rows, dim))

    print('\n' + '=' * 88)
    print('DISTILLATION FIDELITY: val-split L_hall vs the frozen teacher (Hoffman sigmoid-L2,')
    print('the exact training objective). Lower = student activations match teacher.')
    print('=' * 88)
    for src in ('netH_trained', 'netH_random'):
        v = np.array(hall[src])
        print('  %-14s %.6f +/- %.6f  (n=%d test tiles)' % (src, v.mean(), v.std(), len(v)))
    tr, rd = np.array(hall['netH_trained']).mean(), np.array(hall['netH_random']).mean()
    print('  reduction vs random init: %.1f%%' % (100.0 * (rd - tr) / rd if rd > 0 else float('nan')))

    with open(opt.out_csv, 'w') as f:
        f.write('layer,source,dim,rows_fit,lambda,r2,r2_vv,r2_vh,r2_diff\n')
        for r in rows:
            f.write('%d,%s,%d,%d,%g,%.6f,%.6f,%.6f,%.6f\n'
                    % (r['layer'], r['source'], r['dim'], r['rows_fit'], r['lam'],
                       r['r2'], r['r2_vv'], r['r2_vh'], r['r2_diff']))
    print('\nPer-layer results written to %s' % opt.out_csv)


if __name__ == '__main__':
    main()
