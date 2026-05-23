from typing import Optional, Tuple, Union

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.local_model import LocalModel
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.retry_policy import create_retry_decorator


# Custom LocalModel that properly supports structured outputs with vllm
class StructuredLocalModel(LocalModel):
    """LocalModel subclass that properly uses vllm's structured output features.

    This adds the response_format parameter when a schema is requested, which tells
    vllm to enforce JSON schema compliance via guided decoding.
    """

    @create_retry_decorator(PS.LOCAL)
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = prompt

        client = self.load_model(async_mode=False)

        # Build request kwargs
        kwargs = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            **self.generation_kwargs,
        }

        # If schema is provided, add response_format for structured outputs
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True
                }
            }

        response: ChatCompletion = client.chat.completions.create(**kwargs)
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

    @create_retry_decorator(PS.LOCAL)
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = prompt

        client = self.load_model(async_mode=True)

        # Build request kwargs
        kwargs = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            **self.generation_kwargs,
        }

        # If schema is provided, add response_format for structured outputs
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True
                }
            }

        response: ChatCompletion = await client.chat.completions.create(**kwargs)
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0
