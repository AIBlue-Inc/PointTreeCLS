from .runner import run_net
from .runner import test_net
from .runner_BERT_pretrain import run_net as BERT_pretrain_run_net
from .runner_BERT_finetune import run_net as BERT_finetune_run_net
from .runner_BERT_finetune import test_net as BERT_test_run_net

from .runner_GPT_pretrain import run_net as GPT_pretrain_run_net
from .runner_GPT_finetune import run_net as GPT_finetune_run_net
from .runner_GPT_finetune import test_net as GPT_test_run_net