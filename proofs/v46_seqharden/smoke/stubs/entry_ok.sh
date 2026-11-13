#!/bin/bash
touch "${SMK_MARK:?}/entry_ok.started"; echo "entry stub ok (GA_START_UTC=${GA_START_UTC:?} WRF_GPU2_SEQ_SMOKE=${WRF_GPU2_SEQ_SMOKE:-0})"; exit 0
