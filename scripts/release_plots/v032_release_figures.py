"""Readable release plots from accepted VAL32 / sweep24 fields, CPU only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    d=json.loads((ROOT/'docs/release/evidence/v032/release_summary.json').read_text())
    timing,parallel=d['prod_timing'],d['parallel']
    board=parallel['board_wh_per_case_fc_h'];host=(200*4/12)*parallel['whole_s_per_case_fc_h']/3600
    for dark in (False,True):
        bg,fg,muted=('#111827','#eef3fa','#b8c4d7') if dark else ('#fff','#17243b','#51627a')
        suffix='_dark' if dark else ''
        with plt.rc_context({'font.family':'DejaVu Sans','font.size':16,'text.color':fg,
                             'axes.labelcolor':fg,'xtick.color':fg,'ytick.color':fg}):
            for kind in ('speed','parallel','energy'):
                fig,ax=plt.subplots(figsize=(12,6.4),dpi=170);fig.set_facecolor(bg);ax.set_facecolor(bg)
                for spine in ax.spines.values():spine.set_visible(False)
                ax.set_axisbelow(True);ax.grid(axis='x',color=muted,alpha=.18);ax.tick_params(length=0)
                if kind=='speed':
                    labels=['CPU-WRF · 12 cores','v0.3.1 · matched host','v0.3.2 · matched host']
                    values=[92.3,timing['baseline_matched_s_per_fc_h'],timing['absolute_whole_s_per_fc_h']]
                    ax.barh(range(3),values,color=['#94a3b8','#507cc0','#18a58b'],height=.56)
                    for i,v in enumerate(values):
                        text=f'{v:.3f} s [M]' if i else '92.3 s [M]'
                        if i==2:text+=f' · {92.3/v:.1f}× CPU'
                        ax.text(v+1.5,i,text,va='center',weight='bold')
                    ax.set_xlim(0,116);title='Canary 9/3 km · complete 24-hour process'
                    xlabel='Seconds per forecast hour · lower is faster'
                    foot='Matched whole-run B/A = 0.901 (−9.9%) [M]. Host contention; absolute rates conservative.\nGPU includes init/output/completion; CPU reference is main-loop. Not a quiet absolute benchmark.'
                elif kind=='parallel':
                    labels=['CPU-WRF · 3 × 4 cores','RTX 5090 · N=4 whole','RTX 5090 · N=4 steady']
                    values=[123.4,parallel['whole_s_per_case_fc_h'],parallel['steady_s_per_case_fc_h']]
                    ax.barh(range(3),values,color=['#94a3b8','#18a58b','#167cbb'],height=.56)
                    for i,v in enumerate(values):ax.text(v+1.8,i,f'{v:.3f} s [M]'+(f' · {123.4/v:.1f}× CPU' if i else ''),va='center',weight='bold')
                    ax.set_xlim(0,170);title='Tenerife 9/3/1 km · four independent 24-hour forecasts'
                    xlabel='Aggregate seconds per case-forecast-hour · lower is faster'
                    foot='Whole: first start → last output / (4 × 24 h), init/load included; no compression.\nSteady: measured all-four-running window. Longer runs amortize fixed start-up per case.'
                else:
                    labels=['CPU-WRF · 3 × 4 cores','RTX 5090 · N=4 board + host']
                    cpu=200*123.4/3600;values=[cpu,board+host]
                    ax.barh(0,cpu,color='#94a3b8',height=.56,label='CPU power × clock [I]')
                    ax.barh(1,board,color='#18a58b',height=.56,label='GPU board integral [M]')
                    ax.barh(1,host,left=board,color='#507cc0',hatch='///',height=.56,label='GPU host share [I]')
                    for i,v in enumerate(values):ax.text(v+.13,i,f'{v:.3f} Wh [I]',va='center',weight='bold')
                    ax.set_xlim(0,9.5);ax.legend(loc='center left',bbox_to_anchor=(.45,.12),fontsize=14,frameon=False)
                    title='Tenerife energy · 24-hour N=4 sweep';xlabel='Wh per case-forecast-hour · lower is less energy'
                    foot=f'Board {board:.3f} Wh [M], E158 actual timestamps; no idle subtraction in headline.\nCPU: owner-reported 200 W; GPU host share: 4/12 × 200 W [I]. No whole-node energy claim.'
                ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel(xlabel,labelpad=14)
                fig.suptitle(title,x=.025,y=.975,ha='left',fontsize=20,weight='bold')
                fig.subplots_adjust(left=.34 if kind=='energy' else .29,right=.98,top=.86,bottom=.25)
                fig.text(.025,.04,foot,fontsize=13,color=muted,linespacing=1.6)
                fig.savefig(ROOT/f'docs/release/img/v032_{kind}{suffix}.png',facecolor=bg);plt.close(fig)


if __name__=='__main__':main()
