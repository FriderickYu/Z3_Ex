import argparse
import os
from api_key.llm_dispatcher import LLMDispatcher
from rules.rules_pooling import RulesPooling
from utils.prompt_loader import load_prompt_template
from core.rule_sampler import RuleSampler
from core.prompt_builder import PromptBuilder
from core.llm_parser import LLMParser
from core.sample_writer import SampleWriter
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dispatcher = LLMDispatcher(model_name=args.model_name, api_key_path=args.api_key_path)
    rule_pool = RulesPooling()
    prompt_template = load_prompt_template(args.prompt_path)

    sampler = RuleSampler(rule_pool)
    builder = PromptBuilder(prompt_template)
    caller = LLMParser(dispatcher)
    writer = SampleWriter(args.output)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("Generating samples", total=args.num_samples)
        generated = 0
        attempts = 0
        max_attempts = args.num_samples * 10

        while generated < args.num_samples and attempts < max_attempts:
            attempts += 1
            sampled = sampler.sample_and_validate(len(rule_pool.available_rules))
            if sampled:
                rules, z3_exprs = sampled
                prompt = builder.build(z3_exprs)
                result = caller.call_and_parse(prompt)
                if result:
                    writer.write_sample(result)
                    generated += 1
                    progress.advance(task)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Logical Reasoning Sample Generator")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output", type=str, default="output/samples.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompt/lsat_prompt.txt")
    parser.add_argument('--model_name', type=str, default="deepseek-chat", help="Name of the LLM model (e.g., gpt-4, deepseek-chat)")
    parser.add_argument("--api_key_path", type=str, default="api_key/ds-api_key.txt")

    args = parser.parse_args()
    main(args)