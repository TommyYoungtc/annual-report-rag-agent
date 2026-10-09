from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class PromptVariant(str, Enum):
    FULL = "full"
    COMPRESSED = "compressed"


FULL_SYSTEM_PROMPT = """你是上市公司年报研究助手。只能依据提供的年报证据回答，不能使用记忆补全。
回答前逐项检查公司、报告年份、年度或季度口径、指标名称、单位和业务范围是否与问题一致。
如果证据缺少任何必要约束，必须拒绝作答，不能根据相似指标猜测。
涉及跨年份比较时，必须分别找到每个年份的证据；涉及计算时使用证据中的原始数值和单位进行计算。
引用必须来自实际支持答案的证据，并保留 chunk_id 与页码。不要引用只是在主题上相似的段落。
最终仅输出 JSON，字段为 status、answer、reason、citations、calculation。status 只能是 answered 或 refused；citations 是包含 chunk_id、page、quote 的数组。"""


COMPRESSED_SYSTEM_PROMPT = """仅据给定年报证据答题；核对公司/年份/年度季度/指标/单位/范围，缺一则拒答。跨年须逐年有证据，计算保留原值和单位。只引支持答案的 chunk_id、page、quote。仅输出 JSON：status(answered|refused),answer,reason,citations,calculation。"""


@dataclass(frozen=True, slots=True)
class PromptBuild:
    variant: PromptVariant
    messages: tuple[dict[str, str], ...]
    system_prompt: str
    prompt_sha256: str


def build_prompt(
    query: str,
    context: str,
    *,
    variant: PromptVariant,
) -> PromptBuild:
    system_prompt = (
        FULL_SYSTEM_PROMPT if variant == PromptVariant.FULL else COMPRESSED_SYSTEM_PROMPT
    )
    user_prompt = f"问题：{query}\n\n证据：\n{context or '[无可用证据]'}"
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return PromptBuild(
        variant=variant,
        messages=(
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ),
        system_prompt=system_prompt,
        prompt_sha256=digest,
    )
