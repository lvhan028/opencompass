# flake8: noqa
# yapf: disable
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from lmdeploy.archs import autoget_backend

from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from .huggingface_above_v4_33 import (_convert_chat_messages,
                                      _format_with_fast_chat_template,
                                      _get_meta_template,
                                      _get_possible_max_seq_len)

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


class TurboMindModelwithChatTemplate(BaseModel):
    def __init__(
        self,
        path: str,
        tokenizer_only: bool = False,
        backend: str = 'auto',
        engine_config: Dict = {},
        gen_config: Dict = {},
        max_seq_len: int = None,
        meta_template: Optional[Dict] = None,
        fastchat_template: Optional[str] = None,
        stop_words: List[str] = [],
    ):
        from lmdeploy.version import version_info
        from transformers import AutoTokenizer

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)

        self.origin_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if not tokenizer_only:
            DEFAULT_ENGING_CONFIG = {'session_len': self.max_seq_len}
            _engine_config = DEFAULT_ENGING_CONFIG.copy()
            _engine_config.update(engine_config)
            self._build_engine(path, backend, _engine_config)
        self.gen_config = gen_config
        self.version_info = version_info
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        self.logger.info(f'using stop words: {self.stop_words}')

    def _build_engine(self, model_path, backend, engine_config: Dict):
        if backend == 'auto':
            backend = autoget_backend(model_path)
        assert backend in ['turbomind', 'pytorch'], f'unexpected backend {backend}'
        if backend == 'turbomind':
            from lmdeploy import TurbomindEngineConfig
            from lmdeploy.turbomind import TurboMind
            engine_config = TurbomindEngineConfig(**engine_config)
            engine = TurboMind.from_pretrained(model_path, engine_config=engine_config)
        else:
            from lmdeploy import PytorchEngineConfig
            from lmdeploy.pytorch.engine import Engine
            engine_config = PytorchEngineConfig(**engine_config)
            engine = Engine(model_path=model_path, engine_config=engine_config)
        self.backend = backend
        self.engine_config = engine.engine_config
        self.generators = [engine.create_instance() for i in range(engine.engine_config.max_batch_size)]
        self.tokenizer = engine.tokenizer

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.origin_tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.origin_tokenizer.decode(token_id))
        if self.origin_tokenizer.eos_token is not None:
            potential_stop_words.append(self.origin_tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def generate(self,
                 inputs: List[str],
                 max_out_len: int = 512,
                 stopping_criteria: List[str] = [],
                 do_sample: Optional[bool] = None,
                 temperature: int = 1,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of prompts
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        assert isinstance(inputs, List), f'List(str) is expected, but got {type(inputs)}'

        messages = _convert_chat_messages(inputs)
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
        else:
            messages = [self.origin_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]

        stop_words = list(set(self.stop_words + stopping_criteria))
        encode_stop_words = []
        if stop_words is not None and len(stop_words) > 0:
            for words in stop_words:
                encode_stop_words += self.tokenizer.encode(words, add_bos=False)

        DEFAULT_GEN_CONFIG = {
            'max_new_tokens': max_out_len,
            'min_new_tokens': 1,
            'top_k': 1,
            'stop_words': encode_stop_words,
        }

        gen_config = copy.deepcopy(DEFAULT_GEN_CONFIG)
        gen_config.update(self.gen_config)
        if do_sample:
            gen_config['top_k'] = 50
            gen_config['temperature'] = temperature

        from lmdeploy.messages import GenerationConfig
        gen_config = GenerationConfig(**gen_config)
        if self.version_info >= (0, 6, 0):
            gen_config.stop_words = stop_words

        results = []
        # inference with batch
        batch_size = self.engine_config.max_batch_size
        for i in range(0, len(messages), batch_size):
            batch_message = messages[i:i+batch_size]
            n = len(batch_message)
            session_ids = [i for i in range(n)]
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                _results = list(
                    executor.map(
                        self._generate,
                        self.generators[:n],
                        session_ids,
                        batch_message,
                        [gen_config] * n,
                    ))
                results += _results

        for s in stop_words:
            results = [r.split(s)[0] for r in results]
        return results

    def _generate(self,
                  generator,
                  session_id,
                  prompt: PromptType,
                  gen_config=None) -> str:
        """Generate results given a list of inputs.

        Args:
            prompt (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            gen_config (GenerationConfig, optional): Generation
                config to set arguments like top_k, top_p, temperature.
        Returns:
            str: The generated string.
        """
        assert type(prompt) is str, 'We only support string for TurboMind Python API'

        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        if self.backend == 'turbomind':
            for outputs in generator.stream_infer(session_id=session_id,
                                                  input_ids=input_ids,
                                                  gen_config=gen_config,
                                                  sequence_start=True,
                                                  sequence_end=True,
                                                  step=0,
                                                  stream_output=False):
                if self.version_info >= (0, 4, 0):
                    output_ids = outputs.token_ids
                else:
                    _, output_ids, _ = outputs
        else:
            if self.version_info >= (0, 4, 0):
                outputs = generator.infer(session_id,
                                          input_ids,
                                          gen_config=gen_config)
                output_ids = outputs.token_ids
            else:
                _, output_ids, _ = generator.infer(session_id,
                                                   input_ids,
                                                   gen_config=gen_config)
            generator.end(session_id)
        response = self.tokenizer.decode(output_ids)
        response = valid_str(response)
        return response

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        m = _convert_chat_messages([prompt])[0]
        t = self.origin_tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])
