
python -m scripts.paddle_runner `
  --serial e5081c2a `
  --combat-macro "recordings/mission12_01.json" `
  --combat-macro-loops 1 `
  --max-combat 1 `
  --cfg "configs/ocr_states_fsm.json5" `
  --det-dir "E:\.paddleocr\whl\det\ch_PP-OCRv4_det_infer" `
  --rec-dir "E:\.paddleocr\whl\rec\ch_PP-OCRv4_rec_infer" `
  --cls-dir "E:\.paddleocr\whl\ch_ppocr_mobile_v2.0_cls_infer" `
  --prestart-macro `
  --prestart-delay 0.0 `
  --interval 0.5

Read-Host "Press Enter to exit"