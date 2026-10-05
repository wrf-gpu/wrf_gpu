"""Separate, readable release plots from retained receipt fields and samples."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / 'docs/release/evidence/v031'
OUT = ROOT / 'docs/release/img'


def main():
    summary = json.loads((EV / 'summary.json').read_text())
    receipt = json.loads((EV / 'prod24_receipt_selected.json').read_text())
    sweep = json.loads((EV / 'parallel_sweep_selected.json').read_text())
    energy = json.loads((EV / 'energy_samples.json').read_text())
    gpu = receipt['t_end'] / 24
    for dark in (False, True):
        bg, fg, muted = ('#111827', '#eef3fa', '#b8c4d7') if dark else ('#ffffff', '#17243b', '#51627a')
        colors = ['#94a3b8', '#507cc0', '#18a58b', '#167cbb']
        suffix = '_dark' if dark else ''
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 16,
                             'text.color': fg, 'axes.labelcolor': fg,
                             'xtick.color': fg, 'ytick.color': fg}):
            for kind in ('speed', 'parallel', 'energy'):
                fig, ax = plt.subplots(figsize=(12, 6.4), dpi=170)
                fig.set_facecolor(bg); ax.set_facecolor(bg)
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.tick_params(length=0); ax.set_axisbelow(True)
                ax.grid(axis='x', color=muted, alpha=.18)
                if kind == 'speed':
                    names = ['CPU-WRF · 12 cores', 'v0.3.0 · RTX 5090', 'v0.3.1 · RTX 5090']
                    values = [92.3, 4.126, gpu]
                    ax.barh(range(3), values, color=colors[:3], height=.56)
                    for i, value in enumerate(values):
                        label = '92.3 s [M]' if i == 0 else f'{value:.3f} s [M]'
                        if i == 2: label += f' · {92.3/gpu:.1f}× CPU'
                        ax.text(value + 1.5, i, label, va='center', weight='bold')
                    title = 'Canary 9/3 km · complete cached 24-hour forecast'
                    xlabel = 'Seconds per forecast hour · lower is faster'
                    note = 'GPU: process start → completion, init and output included. CPU: main-loop reference.\nRTX 5090 / Ryzen 9 9950X; stated clock ratio 25.6×. Receipts: docs/release/V0.3.1.md'
                    ax.set_xlim(0, 116)
                elif kind == 'parallel':
                    names = ['CPU-WRF · 3 × 4 cores', 'RTX 5090 · N=1', 'RTX 5090 · N=3', 'RTX 5090 · N=4']
                    values = [123.4] + [sweep['phases'][n]['aggregate']['aggregate_s_per_case_fc_h'] for n in ('n1', 'n3', 'n4')]
                    ax.barh(range(4), values, color=colors, height=.56)
                    for i, value in enumerate(values):
                        ax.text(value + 1.8, i, f'{value:.2f} s [M]', va='center', weight='bold')
                    title = 'Tenerife 9/3/1 km · independent cases on one GPU'
                    xlabel = 'Aggregate seconds per case-forecast-hour · lower is faster'
                    note = 'Six-hour runs include each case’s init/load. First start → last history file, divided by N × 6 h.\nNo compression (v0.3.0 used it). No 24 h extrapolation. N=4: 15.9× CPU throughput [M].'
                    ax.set_xlim(0, 154)
                else:
                    names = ['CPU-WRF · 3 × 4 cores', 'RTX 5090 · N=1', 'RTX 5090 · N=3', 'RTX 5090 · N=4']
                    cpu = energy['cpu']['j_per_case_forecast_hour'] / 3600
                    board = [energy['phases'][n]['board_j_per_case_forecast_hour'] / 3600 for n in ('n1','n3','n4')]
                    host = [energy['phases'][n]['host_share_j_per_case_forecast_hour'] / 3600 for n in ('n1','n3','n4')]
                    ax.barh(0, cpu, color=colors[0], height=.56, label='CPU power × clock [I]')
                    ax.barh([1,2,3], board, color=colors[2], height=.56, label='GPU board integral [M]')
                    ax.barh([1,2,3], host, left=board, color=colors[1], hatch='///', height=.56, label='GPU host share [I]')
                    values = [cpu] + [b+h for b,h in zip(board,host)]
                    for i, value in enumerate(values): ax.text(value + .13, i, f'{value:.2f} Wh [I]', va='center', weight='bold')
                    ax.legend(loc='center left', bbox_to_anchor=(.48, .22), fontsize=14, frameon=False)
                    title = 'Tenerife energy · measured board + inferred host share'
                    xlabel = 'Wh per case-forecast-hour · lower is less energy'
                    note = 'CPU: owner-reported 200 W × main-loop rate [I]. GPU: E158 timestamp trapezoid [M]; tails excluded.\nHost share: (4/12) × 200 W ≈ 67 W [I]. Six-hour sweep; no package/whole-node power claim.'
                    ax.set_xlim(0, 8.3)
                ax.set_yticks(range(len(names)), names); ax.invert_yaxis()
                ax.set_xlabel(xlabel, labelpad=14)
                fig.suptitle(title, x=.025, y=.975, ha='left', fontsize=20, weight='bold')
                fig.subplots_adjust(left=.29, right=.98, top=.86, bottom=.25)
                fig.text(.025, .04, note, fontsize=13, color=muted, linespacing=1.6)
                fig.savefig(OUT / f'v031_{kind}{suffix}.png', facecolor=bg)
                plt.close(fig)


if __name__ == '__main__':
    main()
