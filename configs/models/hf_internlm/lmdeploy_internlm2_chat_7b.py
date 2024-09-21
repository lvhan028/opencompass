from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internlm2-chat-7b-turbomind',
        path='internlm/internlm2-chat-7b',
        backend='auto',
        engine_config=dict(session_len=7168, max_batch_size=256, tp=1),
        gen_config=dict(do_sample=False, max_new_tokens=1024),
        max_seq_len=7168,
        max_out_len=1024,
        batch_size=5000,
        run_cfg=dict(num_gpus=1),
    )
]
