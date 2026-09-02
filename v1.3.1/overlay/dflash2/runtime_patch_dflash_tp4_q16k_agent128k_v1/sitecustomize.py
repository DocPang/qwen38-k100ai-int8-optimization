"""TP4 DFlash2 Agent128K q16K experimental composition.

Model interpreters load the already-qualified TP4 DFlash2 block8/full-window
stack first, then install the q16K long-prefill attention addon. Helper
interpreters bypass the entire model patch stack.
"""
from __future__ import annotations
import os, runpy

_DROOT='/data/qwen38-dflash2-k100ai'
_TROOT='/data/qwen38-27b-k100ai-int8-opt'
try:
    _cmd=open('/proc/self/cmdline','rb').read().replace(b'\0',b' ').decode('utf-8','replace')
except Exception:
    _cmd=''
_helper=(
    ('cdll.LoadLibrary' in _cmd and 'torchinductor_tp4' in _cmd)
    or 'multiprocessing.resource_tracker' in _cmd
    or '/usr/local/bin/ninja' in _cmd
    or ' ninja --version' in _cmd
)
if _helper:
    print(f'[K100 DFlash2 TP4 q16K helper bypass] pid={os.getpid()}',flush=True)
else:
    runpy.run_path(f'{_DROOT}/runtime_patch_dflash_tp4_agent128k_v1/sitecustomize.py',run_name='__q38_dflash_tp4_q16_base__')
    runpy.run_path(f'{_TROOT}/runtime_patch_sglang_tp4_u036_q16k_v1/sitecustomize.py',run_name='__q38_tp4_q16_attention_addon__')

    _selector_topk=int(os.environ.get('SGLANG_DFLASH2_SELECTOR_TOPK_OVERRIDE','16') or 16)
    if _selector_topk not in (16,32,64):
        raise RuntimeError(f'DFlash2 q16 selector top-k override must be one of 16/32/64, got {_selector_topk}')
    if _selector_topk != 16:
        from sglang.srt.models.dflash import DFlash2DraftModel as _DFlash2DraftModel
        _orig_dflash2_init=_DFlash2DraftModel.__init__
        def _q16_dflash2_init(self,*args,**kwargs):
            _orig_dflash2_init(self,*args,**kwargs)
            trained=int(getattr(self.candidate_selector,'top_k',-1))
            if trained != 16:
                raise RuntimeError(f'DFlash2 q16 top-k override expected trained top_k=16, got {trained}')
            self.candidate_selector.top_k=_selector_topk
            print(f'[K100 DFlash2 selector top-k] override trained=16 runtime={_selector_topk}',flush=True)
        _DFlash2DraftModel.__init__=_q16_dflash2_init
    print(f'[K100 DFlash2 TP4 q16K] composition installed: base DFlash2 + q16K prefill addon; selector_topk={_selector_topk}',flush=True)
